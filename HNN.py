import argparse, os, json
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import csv
from torch.utils.data import TensorDataset, DataLoader

# Utilities
def seed_all(seed=420):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def build_mlp(in_dim: int, out_dim: int, hidden: int = 256, depth: int = 3, act: nn.Module = nn.SiLU()) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden), act]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), act]
    layers += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)

def load_npy(path):
    arr = np.load(path)
    if arr.ndim == 1:
        arr = arr[None, :]
    return torch.from_numpy(arr).float()

def fit_norm(z: torch.Tensor):
    mean = z.mean(0)
    std  = z.std(0).clamp_min(1e-8)
    return mean, std

def _to_tensor(x):
    arr = np.array(x, dtype=np.float64 if np.asarray(x).dtype==np.float64 else np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return torch.from_numpy(arr).float()

def load_json_data(path: str):
    with open(path, 'r') as f:
        data = json.load(f)
    out = {}
    # Accept aliases X->z, Y->dz, y->z_next
    if 'z' in data or 'X' in data:
        out['z'] = _to_tensor(data.get('z', data.get('X')))
    if 'X' in data:                                # <-- add this
        out['X'] = _to_tensor(data['X'])           # <-- add this
    if 'dz' in data or 'Y' in data:
        out['dz'] = _to_tensor(data.get('dz', data.get('Y')))
    if 'z_next' in data or 'y' in data:
        out['z_next'] = _to_tensor(data.get('z_next', data.get('y')))
    if 'z0' in data:
        out['z0'] = _to_tensor(data['z0'])
    if 'dt' in data:
        out['dt'] = float(data['dt'])
    return out

# Model (Greydanus 2019 + CHNN 2020)
@dataclass
class HNNConfig:
    n_bodies: int
    hidden: int = 256
    depth: int = 3
    separable: bool = True
    learn_mass: bool = True
    tie_body_mass: bool = True  # tie x/y/z per body

class HNN(nn.Module):
    """
    Hamiltonian Neural Network (Greydanus et al., 2019) with optional holonomic constraints (Finzi et al., 2020).
    - Separable H(q,p) = T(p) + V_theta(q) with learned masses (default), or general scalar H_theta(q,p).
    - Time derivatives from autograd: dz/dt = J grad_z H, with canonical J.
    - Holonomic constraints g(q) = 0 enforced via tangent-space projection + Lagrange multipliers.
    """
    def __init__(
        self,
        n_bodies: int,
        hidden: int = 256,
        depth: int = 3,
        constraint_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        separable: bool = True,
        learn_mass: bool = True,
        tie_body_mass: bool = True,
    ):
        super().__init__()
        self.n_bodies = n_bodies
        self.ndof = 3 * n_bodies
        self.constraint_fn = constraint_fn
        self.separable = separable
        self.learn_mass = learn_mass
        self.tie_body_mass = tie_body_mass
        self.depth = depth  # <-- make sure this exists if you reference it

        # Normalization buffers
        self.register_buffer("z_mean", torch.zeros(2 * self.ndof))
        self.register_buffer("z_std",  torch.ones(2 * self.ndof))

        if separable:
            self.n_pairs = n_bodies * (n_bodies - 1) // 2
            pair_idx = torch.triu_indices(n_bodies, n_bodies, 1)
            self.register_buffer("pair_idx", pair_idx)  # [2, n_pairs]

            # shared per-pair MLP φ: R -> R, applied per pair then summed
            self.phi_pair = build_mlp(
                in_dim=1, out_dim=1,
                hidden=max(16, hidden // 2),
                depth=max(2, depth - 1)
            )
        else:
            self.mlp = build_mlp(
                in_dim=2 * self.ndof, out_dim=1,
                hidden=hidden, depth=depth
            )

        if learn_mass:
            if tie_body_mass:
                self.log_m_body = nn.Parameter(torch.zeros(n_bodies))      # [N]
            else:
                self.log_m_dof  = nn.Parameter(torch.zeros(3 * n_bodies))  # [3N]
        else:
            # fixed unit masses (buffers, not trainable)
            self.register_buffer("m_body_fixed", torch.ones(n_bodies))

        # Canonical symplectic form J = [[0, I], [-I, 0]]
        D = torch.get_default_dtype()
        J = torch.zeros(2 * self.ndof, 2 * self.ndof, dtype=D)
        I = torch.eye(self.ndof, dtype=D)
        J[:self.ndof, self.ndof:] = I
        J[self.ndof:, :self.ndof] = -I
        self.register_buffer("J", J)
        self.register_buffer("softening_eps", torch.tensor(0.1))  # default; override if you want

    # Energies
    def kinetic(self, p: torch.Tensor) -> torch.Tensor:
        # p: [B, ndof], ndof=3*N
        if self.learn_mass:
            if hasattr(self, "log_m_body"):
                m_vec = self.log_m_body.exp().repeat_interleave(3)      # [3N]
            else:
                m_vec = self.log_m_dof.exp()                            # [3N]
        else:
            m_vec = torch.ones(self.ndof, device=p.device, dtype=p.dtype)

        Minv = 1.0 / (m_vec + 1e-8)                                     # [3N]
        return 0.5 * (p.pow(2) * Minv.unsqueeze(0)).sum(dim=1, keepdim=True)

    def potential(self, q: torch.Tensor) -> torch.Tensor:
        B = q.size(0)
        q_b = q.view(B, self.n_bodies, 3)
        qi = q_b[:, self.pair_idx[0]]
        qj = q_b[:, self.pair_idx[1]]

        eps2 = float(self.softening_eps.item()) ** 2
        d = torch.sqrt(((qi - qj) ** 2).sum(-1) + eps2)     # [B, n_pairs]

        # Choose one input to phi:  1/d  (simple & works well)
        s = (1.0 / d).unsqueeze(-1)                         # [B, n_pairs, 1]

        # if not self.learn_mass:
        #     m = self.m_body_fixed                          # [N]
        # elif hasattr(self, "log_m_body"):
        #     m = self.log_m_body.exp()                      # [N]
        # else:
        #     m = self.log_m_dof.view(self.n_bodies, 3).exp().mean(-1)  # [N]
        # mi = m[self.pair_idx[0]]; mj = m[self.pair_idx[1]]             # [n_pairs]
        # mass_pair = (mi * mj).unsqueeze(0).unsqueeze(-1)               # [1,n_pairs,1]
        # s = (mass_pair / d.unsqueeze(-1))                               # [B,n_pairs,1]

        v_ij = self.phi_pair(s).squeeze(-1)                 # [B, n_pairs]
        return v_ij.sum(dim=1, keepdim=True)                # [B, 1]


    def hamiltonian(self, z: torch.Tensor) -> torch.Tensor:
        if self.separable:
            q, p = torch.split(z, [self.ndof, self.ndof], dim=1)
            H = self.kinetic(p) + self.potential(q)     # [B,1]
            return H.squeeze(-1)                         # [B]
        else:
            zn = (z - self.z_mean) / (self.z_std + 1e-8)
            return self.mlp(zn).squeeze(-1)             # [B]

    # Dynamics
    def time_derivatives(self, z: torch.Tensor) -> torch.Tensor:
        # Classic HNN: dz/dt = J grad_z H
        with torch.enable_grad():
            # Ensure z participates in autograd
            if not z.requires_grad:
                z = z.clone().requires_grad_(True)

            H = self.hamiltonian(z)                           # [B]
            (grad,) = torch.autograd.grad(H.sum(), z, create_graph=True)  # [B, 2*ndof]

            # Make sure J matches dtype/device of grad
            J = self.J.to(dtype=grad.dtype, device=grad.device)

            dzdt = grad @ J.t()                               # [B, 2*ndof]
            dqdt, dpdt = dzdt.split(self.ndof, dim=1)

            # Apply holonomic constraints if provided
            if self.constraint_fn is not None:
                dqdt, dpdt = self._apply_constraints(z, dqdt, dpdt)

            # No manual COM/momentum patch here; conservation should emerge from symmetry
            return torch.cat([dqdt, dpdt], dim=1)
        
    def _minv_vec(self, device=None, dtype=None):
        """
        Returns a length-[ndof] tensor with the inverse mass for each coordinate
        (x,y,z for each body). Handles both tied and untied mass parameters.
        """
        device = device if device is not None else self.J.device
        dtype  = dtype  if dtype  is not None else self.J.dtype

        if not self.learn_mass:
            m_per_body = self.m_body_fixed.to(device=device, dtype=dtype)        # [N]
            m_per_dof  = m_per_body.repeat_interleave(3)                          # [3N]
            return 1.0 / (m_per_dof + 1e-8)
        else:
            if hasattr(self, "log_m_body"):
                m_per_body = self.log_m_body.exp().to(device=device, dtype=dtype)          # [N]
            else:
                # Untied per-dof (see Fix #3); shape [3N]
                m_per_dof = self.log_m_dof.exp().to(device=device, dtype=dtype)            # [3N]
                return (1.0 / (m_per_dof + 1e-8))

        # Tie mass across x,y,z for each body -> repeat 3 times
        m_per_dof = m_per_body.repeat_interleave(3)                                         # [3N]
        return (1.0 / (m_per_dof + 1e-8))

    def _apply_constraints(self, z: torch.Tensor, dqdt: torch.Tensor, dpdt: torch.Tensor):
        """
        Mass-aware holonomic constraints:
        Project velocities into the tangent space and correct forces using the
        Gauss principle with A = J M^{-1} J^T and M^{-1} weighting.
        """
        B = z.size(0)
        ndof = self.ndof

        # q for autodiff of constraints
        q = z[:, :ndof].detach().requires_grad_(True)
        g = self.constraint_fn(q.view(B, self.n_bodies, 3))  # [B, k] or [B, 1]
        if g.ndim == 1:
            g = g.unsqueeze(1)
        k = g.size(1)

        # Build J_q = ∂g/∂q : [B, k, ndof]
        rows = []
        for i in range(k):
            gi = g[:, i].sum()
            Ji = torch.autograd.grad(gi, q, retain_graph=True, allow_unused=False)[0]  # [B, ndof]
            rows.append(Ji)
        Jq = torch.stack(rows, dim=1)  # [B, k, ndof]

        # Inverse mass per dof, broadcast to batch
        Minv_vec = self._minv_vec(device=z.device, dtype=z.dtype)       # [ndof]
        Minv     = Minv_vec.unsqueeze(0).expand(B, -1)                   # [B, ndof]

        eye_k = torch.eye(k, device=z.device, dtype=z.dtype).unsqueeze(0)  # [1,k,k]

        # A = J M^{-1} J^T
        # (J * Minv) applies Minv along the last dim of J
        A = torch.bmm(Jq * Minv.unsqueeze(1), Jq.transpose(1, 2)) + 1e-6 * eye_k  # [B,k,k]

        # Velocity projection (remove normal component)
        # rhs_v = J M^{-1} v  where v = dqdt
        rhs_v = torch.bmm(Jq * Minv.unsqueeze(1), dqdt.unsqueeze(-1)).squeeze(-1)  # [B,k]
        try:
            L = torch.linalg.cholesky(A)                                # [B,k,k]
            lam_v = torch.cholesky_solve(rhs_v.unsqueeze(-1), L).squeeze(-1)
        except RuntimeError:
            lam_v = torch.linalg.solve(A, rhs_v)

        # dqdt ← dqdt - M^{-1} J^T λ_v
        dqdt = dqdt - (Jq.transpose(1, 2) @ lam_v.unsqueeze(-1)).squeeze(-1) * Minv  # [B, ndof]

        # Force correction for dpdt (constraint impulses)
        # rhs_f = J M^{-1} a  where a = dpdt (since ṗ = ∂H/∂q + J^T λ, here we just correct)
        rhs_f = torch.bmm(Jq * Minv.unsqueeze(1), dpdt.unsqueeze(-1)).squeeze(-1)  # [B,k]
        try:
            lam_f = torch.cholesky_solve(rhs_f.unsqueeze(-1), L).squeeze(-1)
        except RuntimeError:
            lam_f = torch.linalg.solve(A, rhs_f)

        # dpdt ← dpdt - M^{-1} J^T λ_f
        dpdt = dpdt - (Jq.transpose(1, 2) @ lam_f.unsqueeze(-1)).squeeze(-1) * Minv  # [B, ndof]

        return dqdt, dpdt


# Integration (HNN-ODEs style)
def euler_step(f: Callable[[torch.Tensor], torch.Tensor], z: torch.Tensor, dt: float) -> torch.Tensor:
    return z + dt * f(z)

def midpoint_step(f: Callable[[torch.Tensor], torch.Tensor], z: torch.Tensor, dt: float) -> torch.Tensor:
    return z + dt * f(z + 0.5 * dt * f(z))

def rk4_step(f: Callable[[torch.Tensor], torch.Tensor], z: torch.Tensor, dt: float) -> torch.Tensor:
    k1 = f(z)
    k2 = f(z + 0.5 * dt * k1)
    k3 = f(z + 0.5 * dt * k2)
    k4 = f(z + dt * k3)
    return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def rollout(fstep, f, z0: torch.Tensor, steps: int, dt: float) -> torch.Tensor:
    z = z0.clone()
    traj = torch.empty(steps + 1, *z.shape, device=z.device, dtype=z.dtype)
    traj[0] = z
    for t in range(1, steps + 1):
        z = fstep(f, z, dt)
        traj[t] = z
    return traj

# Training losses (paper-consistent)
def train_epoch_vfield(model: HNN, loader: DataLoader, device: str, 
                       batch_size: int = 1024, lr: float = 1e-3) -> float:
    # Vector-field supervision (Greydanus): minimize ||f_theta(z) - \dot{z}||^2.
    model.train()
    # Use AdamW for better weight decay
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5) 
    total = 0.0
    
    # Loop over the DataLoader
    for xb, yb in loader:
        # <-- Move batch to device inside the loop
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        pred = model.time_derivatives(xb)
        loss = F.mse_loss(pred, yb)
        
        opt.zero_grad(); loss.backward(); opt.step()
        # Accumulate loss, weighted by batch size
        total += loss.item() * xb.size(0) 
        
    # Return average loss over the entire dataset
    return total / len(loader.dataset)

def train_epoch_rollout(model: HNN, loader: DataLoader, dt: float, device: str,
                        batch_size: int = 1024, lr: float = 1e-3, stepper: str = "rk4") -> float:
    step_fn = rk4_step if stepper == "rk4" else leapfrog_step
    model.train()
    # Use AdamW for better weight decay
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5) 
    total = 0.0
    
    # Loop over the DataLoader
    for zb, zb_next in loader:
        # <-- Move batch to device inside the loop
        zb = zb.to(device, non_blocking=True)
        zb_next = zb_next.to(device, non_blocking=True)

        z_pred = step_fn(model.time_derivatives, zb, dt)
        loss = F.mse_loss(z_pred, zb_next)
        
        opt.zero_grad(); loss.backward(); opt.step()
        # Accumulate loss, weighted by batch size
        total += loss.item() * zb.size(0)
        
    # Return average loss over the entire dataset
    return total / len(loader.dataset)

def train_epoch_rolloutK(model: HNN, z_t: torch.Tensor, z_targets: torch.Tensor, dt: float, K: int = 4, batch_size: int = 256, lr: float = 1e-3) -> float:
    # Multi-step free rollout supervision (short horizon), common in ODE-style training.
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    total = 0.0
    N = z_t.size(0)
    for i in range(0, N, batch_size):
        zb = z_t[i:i+batch_size]
        tgt = z_targets[:, i:i+batch_size]  # [K+1,b,D]
        pred = rollout(rk4_step, model.time_derivatives, zb, K, dt)  # [K+1,b,D]
        loss = F.mse_loss(pred, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * zb.size(0)
    return total / N

def energy_loss(model: HNN, z: torch.Tensor, H_true: torch.Tensor) -> torch.Tensor:
    # supervised scalar energy when ground-truth H is known (toy systems).
    return F.mse_loss(model.hamiltonian(z), H_true)

def leapfrog_step(dzdt, z, dt):
    ndof = z.shape[1] // 2
    q, p = z[:, :ndof], z[:, ndof:]
    # half kick
    dpdt = dzdt(z)[:, ndof:]
    p_half = p + 0.5 * dt * dpdt
    # drift
    z_half = torch.cat([q, p_half], dim=1)
    dqdt = dzdt(z_half)[:, :ndof]
    q_new = q + dt * dqdt
    # full kick
    z_new_half = torch.cat([q_new, p_half], dim=1)
    dpdt_new = dzdt(z_new_half)[:, ndof:]
    p_new = p_half + 0.5 * dt * dpdt_new
    return torch.cat([q_new, p_new], dim=1)

def get_stepper(name):
    return rk4_step if name == "rk4" else leapfrog_step

# Diagnostics
@torch.no_grad()
def rel_energy_drift(model: HNN, traj: torch.Tensor) -> torch.Tensor:
    # |H(t)-H(0)| / (|H(0)|+eps)
    H = torch.stack([model.hamiltonian(traj[t]) for t in range(traj.size(0))])  # [T+1,B]
    H0 = H[0]
    return (H - H0).abs() / (H0.abs() + 1e-8)

@torch.no_grad()
def total_linear_momentum(z: torch.Tensor) -> torch.Tensor:
    # Sum_i p_i; z: [B, 2*ndof]. Returns [B,3].
    B, D = z.shape
    ndof = D // 2
    N = ndof // 3
    p = z[:, ndof:].view(B, N, 3)
    return p.sum(dim=1)

@torch.no_grad()
def angular_momentum(z: torch.Tensor) -> torch.Tensor:
    B, D = z.shape; ndof = D//2; N = ndof//3
    q = z[:, :ndof].view(B, N, 3); p = z[:, ndof:].view(B, N, 3)
    return torch.cross(q, p, dim=-1).sum(dim=1)  # [B,3]

# Constraint helpers
def constrain_anchor(radius: float = 1.0) -> Callable[[torch.Tensor], torch.Tensor]:
    # Keep body 0 at fixed distance from origin: ||q_0|| = radius.
    def g(q_bN3: torch.Tensor) -> torch.Tensor:
        r0 = q_bN3[:, 0]  # [B,3]
        return (r0.pow(2).sum(-1).sqrt() - radius).unsqueeze(1)  # [B,1]
    return g

def constrain_pair_distance(i: int, j: int, dist: float) -> Callable[[torch.Tensor], torch.Tensor]:
    # Keep distance between body i and j fixed: ||q_i - q_j|| = dist.
    def g(q_bN3: torch.Tensor) -> torch.Tensor:
        rij = q_bN3[:, i] - q_bN3[:, j]
        return (rij.pow(2).sum(-1).sqrt() - dist).unsqueeze(1)  # [B,1]
    return g

def _append_csv(path, **kw):
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    import csv
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(kw.keys()))
        if new: w.writeheader()
        w.writerow(kw)

# Checkpoint IO
def save_checkpoint(path, model):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, path)

def load_checkpoint(path, model, map_location=None):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["state_dict"], strict=False)

# Main (CLI)
def apply_norm(model, mean, std):
    with torch.no_grad():
        model.z_mean.copy_(mean)
        model.z_std.copy_(std)

def build_constraint(args, n_bodies):
    if args.constraint is None:
        return None
    if args.constraint == "anchor":
        return constrain_anchor(radius=args.anchor_radius)
    if args.constraint == "pair":
        i, j, d = args.pair_i, args.pair_j, args.pair_dist
        assert 0 <= i < n_bodies and 0 <= j < n_bodies and i != j
        return constrain_pair_distance(i, j, d)
    raise ValueError(f"Unknown constraint: {args.constraint}")

def main():
    parser = argparse.ArgumentParser(description="All-in-one paper-faithful HNN trainer/evaluator")
    # Data
    parser.add_argument("--train-json", type=str, default="HNN_train.npz", help="JSON with training data (z, dz or z_next, optional dt)")
    parser.add_argument("--test-json", type=str, default="HNN_test.npz", help="JSON with eval data (z0 or z, optional dt)")
    parser.add_argument("--train-z", type=str, help="Path to training states z.npy [N, D]")
    parser.add_argument("--train-dz", type=str, default=None, help="Path to training vector field dz/dt.npy [N, D] (vfield mode)")
    parser.add_argument("--train-z-next", type=str, default=None, help="Path to next states z_{t+1}.npy [N, D] (rollout mode)")
    parser.add_argument("--val-z0", type=str, default=None, help="Path to eval initial states z0.npy [B, D]")
    parser.add_argument("--dt", type=float, default=1e-3, help="Timestep for RK4 rollout/labels")
    # Model
    parser.add_argument("--n-bodies", type=int, required=False, help="Auto-detected from z if not set")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--nonseparable", action="store_true", help="Use general H_theta(q,p) instead of separable H=T(p)+V(q)")
    parser.add_argument("--no-learn-mass", action="store_true", help="Disable learned masses")
    parser.add_argument("--no-tie-body-mass", action="store_true", help="Do not tie masses across x/y/z per body")
    # Constraint
    parser.add_argument("--constraint", type=str, default=None, choices=["anchor", "pair"])
    parser.add_argument("--anchor-radius", type=float, default=1.0)
    parser.add_argument("--pair-i", type=int, default=0)
    parser.add_argument("--pair-j", type=int, default=1)
    parser.add_argument("--pair-dist", type=float, default=1.0)
    # Train
    parser.add_argument("--mode", type=str, default="vfield", choices=["vfield", "rollout"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=420)
    # Eval
    parser.add_argument("--rollout-steps", type=int, default=150)
    # IO
    parser.add_argument("--save", type=str, default="./hnn.pt")
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    # CLI
    parser.add_argument("--plot-batch-index", type=int, default=0, help="Which batch traj to visualize in overlays")
    parser.add_argument("--integrator", type=str, default="rk4", choices=["rk4","leapfrog"], help="Integrator used for rollouts at eval")
    parser.add_argument("--rollout-K", type=int, default=0, help="If >0, also train a short-horizon K-step objective (in addition to 1-step)")
    parser.add_argument("--metrics-csv", type=str, default="plots/metrics.csv", help="Path to write epoch/test metrics CSV")

    args = parser.parse_args()

    seed_all(args.seed)

    # Define file paths
    train_data_path = "HNN_train.npz"  # <-- Prioritize .npz
    test_data_path = "HNN_test.npz"    # <-- Prioritize .npz
    
    # Load training data
    z = dz = z_next = None  # initialize so they're always defined

    # Load from NPZ first
    if os.path.exists(train_data_path):
        print(f"Loading from {train_data_path}")
        data = np.load(train_data_path)
        # Load as CPU tensors. Batches will be moved to device in the loop.
        z = torch.from_numpy(data['X']).float()        # <-- Stays on CPU
        z_next = torch.from_numpy(data['y']).float()   # <-- Stays on CPU
        args.dt = float(data['dt'])
        args.mode = "rollout" # We know .npz data is for rollout
        print(f"Loaded training NPZ. z: {z.shape}, z_next: {z_next.shape}, dt: {args.dt}")

    # Fallback to JSON
    elif args.train_json and os.path.exists(args.train_json):
        jd = load_json_data(args.train_json)
        if 'dt' in jd:
            args.dt = jd['dt']
        z = jd.get('z')
        dz = jd.get('dz')
        z_next = jd.get('z_next')

        if z is None:
            raise SystemExit("HNN_train.json must contain at least 'z'")

        # Data stays on CPU
        print(f"Loaded training JSON: {args.train_json}")

        if z_next is not None and dz is None and args.mode != "rollout":
            print("Detected next-state labels in training JSON -> switching to rollout training.")
            args.mode = "rollout"
            
    # Fallback to NPY
    else:
        if args.train_z is None:
            raise SystemExit("--train-z is required when no train JSON is provided")
        z = load_npy(args.train_z)
        print(f"Loaded z: {tuple(z.shape)} from {args.train_z}")
        # (One would also load dz or z_next from .npy here if using that format)

    N, D = z.shape
    # Auto-detect n_bodies
    if args.n_bodies is None:
        if D % 6 != 0:
            raise SystemExit(f"Cannot infer n_bodies from z.shape={z.shape}; expected D divisible by 6")
        args.n_bodies = D // 6
        print(f"Auto-detected n_bodies = {args.n_bodies}")
    assert D == 2*3*args.n_bodies, "z dimensionality must match n_bodies"

    # --- Create Train DataLoader ---
    if args.mode == "rollout":
        if z_next is None:
            raise SystemExit("Mode 'rollout' requires 'y' or 'z_next' data.")
        train_dataset = TensorDataset(z, z_next)
    else:
        if dz is None:
            raise SystemExit("Mode 'vfield' requires 'dz' or 'Y' data.")
        train_dataset = TensorDataset(z, dz)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,                                       # <-- Automatically shuffles data each epoch
        num_workers = 0 if os.name == "nt" else 4,          # <-- Tune this (e.g., 0, 2, 4)
        pin_memory=True,                                    # <-- Speeds up CPU-to-GPU transfer
        drop_last=True                                      # <-- Avoids tiny final batches
    )
    print(f"Created Train DataLoader with {len(train_loader)} batches.")

    # Create Val DataLoader (once, before the loop) ---
    val_loader = None
    if os.path.exists(test_data_path):
        print(f"Loading validation data from {test_data_path}")
        data_val = np.load(test_data_path)
        z_val = torch.from_numpy(data_val['X']).float()
        z_next_val = torch.from_numpy(data_val['y']).float()
        
        val_dataset = TensorDataset(z_val, z_next_val)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch * 2, # Can use a larger batch for validation
            shuffle=False,             # No need to shuffle validation data
            num_workers=2,
            pin_memory=True
        )
        print(f"Created Val DataLoader with {len(val_loader)} batches.")
    
    elif args.test_json and os.path.exists(args.test_json):
        # Fallback to loading test JSON for validation
        print(f"Loading validation data from {args.test_json}")
        jd_val = load_json_data(args.test_json)
        z_val = jd_val.get("z"); z_next_val = jd_val.get("z_next")
        if z_val is not None and z_next_val is not None:
            val_dataset = TensorDataset(z_val, z_next_val)
            val_loader = DataLoader(val_dataset, batch_size=args.batch*2, shuffle=False)
            print(f"Created Val DataLoader from JSON.")

    # Auto-enable separable if not explicitly set
    use_separable = not args.nonseparable

    # Build model
    constraint_fn = build_constraint(args, args.n_bodies)
    model = HNN(
        n_bodies=args.n_bodies,
        hidden=args.hidden,
        depth=args.depth,
        constraint_fn=constraint_fn,
        separable=use_separable,
        learn_mass=not args.no_learn_mass,
        tie_body_mass=not args.no_tie_body_mass,
    ).to(args.device)

    # Normalization from training states
    mean, std = fit_norm(z) # <-- This is fine, z is still on CPU
    apply_norm(model, mean.to(args.device), std.to(args.device))

    # Load checkpoint (optional)
    if args.load is not None and os.path.exists(args.load):
        print(f"Loading checkpoint: {args.load}")
        load_checkpoint(args.load, model, map_location=args.device)

    # Training diagnostics (RSME curves & metrics CSV)
    metrics_csv = args.metrics_csv
    os.makedirs(os.path.dirname(metrics_csv) or ".", exist_ok=True)

    train_rmse_hist, val_rmse_hist = [], []
    epochs = args.epochs

    # Training Loop
    for ep in range(1, epochs + 1):
        if args.mode == "rollout":
            # <-- Pass the loader and device, not the giant tensors
            loss = train_epoch_rollout(
                model, train_loader, args.dt, args.device,
                batch_size=args.batch, lr=args.lr,
                stepper=("rk4" if args.integrator == "rk4" else "leapfrog"),
            )
            # loss is now the epoch's avg MSE, so just sqrt it
            rmse_train = np.sqrt(loss)

        else:  # args.mode == "vfield"
            # Pass the loader and device
            loss = train_epoch_vfield(model, train_loader, args.device,
                                      batch_size=args.batch, lr=args.lr)
            # loss is now the epoch's avg MSE, so just sqrt it
            rmse_train = np.sqrt(loss)

        # Efficient Validation Loop
        rmse_val = None
        if val_loader is not None: # <-- Check if we created a val_loader
            model.eval()           # <-- Set model to evaluation mode
            val_loss_total = 0.0
            step_fn = rk4_step if args.integrator == "rk4" else leapfrog_step # <-- Define step_fn here

            with torch.no_grad():
                # Loop over validation batches
                for z_val_b, z_next_val_b in val_loader:
                    # <-- Move validation batches to device
                    z_val_b = z_val_b.to(args.device, non_blocking=True)
                    z_next_val_b = z_next_val_b.to(args.device, non_blocking=True)
                    
                    z_pred_val = step_fn(model.time_derivatives, z_val_b, args.dt)
                    val_loss = F.mse_loss(z_pred_val, z_next_val_b)
                    # Accumulate loss, weighted by batch size
                    val_loss_total += val_loss.item() * z_val_b.size(0)
            
            # Calculate average RMSE over the entire validation dataset
            rmse_val = np.sqrt(val_loss_total / len(val_loader.dataset))
            model.train() # <-- Set model back to train mode

        train_rmse_hist.append(rmse_train)
        val_rmse_hist.append(rmse_val if rmse_val is not None else float('nan'))

        print(f"[{args.mode}] epoch {ep:03d} | loss {loss:.6e} | RMSE_train {rmse_train:.6e}"
            + (f" | RMSE_val {rmse_val:.6e}" if rmse_val else ""))

        with open(metrics_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if ep == 1:
                writer.writerow(["epoch", "loss", "rmse_train", "rmse_val"])
            writer.writerow([ep, loss, rmse_train, rmse_val])

    # Plot RMSE vs Epoch
    plt.figure(figsize=(6,4))
    plt.plot(range(1, epochs+1), train_rmse_hist, label="Train RMSE")
    if any(not np.isnan(v) for v in val_rmse_hist):
        plt.plot(range(1, epochs+1), val_rmse_hist, label="Val RMSE")
    plt.xlabel("Epoch"); plt.ylabel("RMSE"); plt.title("RMSE vs Epochs")
    plt.grid(alpha=0.3); plt.legend(frameon=False); plt.tight_layout()
    plt.savefig("plots/rmse_epochs.png", dpi=150); plt.close()
    print("Saved training curves to plots/rmse_epochs.png")

    # Save
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": vars(args)}, args.save)
    print(f"Saved checkpoint to: {args.save}")

    def _ensure_plots_dir(path="plots"):
        Path(path).mkdir(parents=True, exist_ok=True)
        return path

    def _traj_to_q(traj_tensor, n_bodies):
        """
        traj_tensor: torch.Tensor [T, B, D] with D = 6*n_bodies (q then p), q is 3D/body.
        returns q: np.ndarray [T, B, n_bodies, 3]
        """
        traj = traj_tensor.detach().cpu().numpy()
        T, B, D = traj.shape
        assert D == 6 * n_bodies, f"Expected D=6*n_bodies, got {D} vs {6*n_bodies}"
        q = traj[..., :3*n_bodies]                 # [T, B, 3*n_bodies]
        q = q.reshape(T, B, n_bodies, 3)           # [T, B, n_bodies, 3]
        return q

    def plot_energy_drift(drift_tensor, outdir):
        """
        drift_tensor: torch.Tensor [T, B] relative energy error per step & traj
        Saves: mean±std over trajectories vs. time.
        """
        drift = drift_tensor.detach().cpu().numpy()  # [T, B]
        mean = drift.mean(axis=1)
        std  = drift.std(axis=1)
        t = np.arange(len(mean))

        plt.figure(figsize=(7, 4))
        plt.plot(t, mean, label="mean rel. energy error")
        plt.fill_between(t, np.maximum(0.0, mean-std), mean+std, alpha=0.2, label="±1 std")
        plt.xlabel("step"); plt.ylabel("relative energy error")
        plt.title("Energy drift over time")
        plt.yscale("log")  # usually spans many orders of magnitude
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "energy_drift.png"), dpi=150)
        plt.close()

    # Eval rollout + diagnostics
    test_used = False

    def _try_get_tensor(jd, key, device, dtype):
        if key in jd and jd[key] is not None:
            t = jd[key]
            if not torch.is_tensor(t):
                t = torch.as_tensor(t, device=device, dtype=dtype)
            else:
                t = t.to(device).to(dtype)
            return t
        return None

    def _overlay_true_pred_plots(jd, traj_pred, n_bodies, outdir, device, dtype):
        """
        Try to find a ground-truth multi-step trajectory in jd, align lengths,
        and save true-vs-pred overlays (per body + all bodies).
        """
        # Prefer explicit keys, but fall back to X if it looks like [T, B, D] or [T, D]
        traj_true = _try_get_tensor(jd, "traj_true", device, dtype)
        if traj_true is None:
            traj_true = _try_get_tensor(jd, "traj", device, dtype)

        if traj_true is None:
            X = _try_get_tensor(jd, "X", device, dtype)
            if X is not None and X.ndim >= 2:
                # Heuristic: treat X with 3 dims as [T,B,D] or [T,1,D] (true traj)
                # Treat [T,D] also as a single-trajectory truth
                if X.ndim == 3 and X.shape[-1] == 6 * n_bodies:
                    traj_true = X  # [T, B, D]
                elif X.ndim == 2 and X.shape[-1] == 6 * n_bodies:
                    traj_true = X.unsqueeze(1)  # [T, 1, D]

        if traj_true is None:
            print("No ground-truth multi-step trajectory ('traj_true'/'traj' or 3D 'X') in test JSON; "
                "skipping true vs predicted overlays.")
            return

        # Align lengths
        T_cmp = min(traj_pred.shape[0], traj_true.shape[0])
        traj_pred_cmp = traj_pred[:T_cmp].detach()  # <--- add .detach()
        traj_true_cmp = traj_true[:T_cmp]
        if torch.is_tensor(traj_true_cmp) and traj_true_cmp.requires_grad:
            traj_true_cmp = traj_true_cmp.detach()  # <--- optional, for safety

        # Extract positions as [T,B,n_bodies,3]
        def _to_q(t):  # [T,B,D] with D = 6*n_bodies, q then p
            q = t[..., :3 * n_bodies]
            return q.reshape(T_cmp, -1, n_bodies, 3)

        q_pred = _to_q(traj_pred_cmp).cpu().numpy()
        q_true = _to_q(traj_true_cmp).cpu().numpy()

        # Pick a single trajectory index for clarity (first in batch)
        bidx = 0
        Path(outdir).mkdir(parents=True, exist_ok=True)

        # Get time axis
        T_cmp = q_pred.shape[0]
        t = np.arange(T_cmp)
        
        # Plot position components (x,y,z) vs time for each body
        for b in range(n_bodies):
            plt.figure(figsize=(10, 5))
            
            # Plot x, y, z components
            for d_idx, d_label in enumerate(['x', 'y', 'z']):
                # Plot true component
                plt.plot(t, q_true[:, bidx, b, d_idx], '-',  alpha=0.7, 
                         label=f'true b{b} ({d_label})')
                # Plot predicted component
                plt.plot(t, q_pred[:, bidx, b, d_idx], '--', alpha=0.9, 
                         label=f'pred b{b} ({d_label})')
            
            plt.title(f"Body {b} — True vs Predicted (Position vs. Time)")
            plt.xlabel("Time Step")
            plt.ylabel("Position")
            plt.grid(True, alpha=0.3)
            plt.legend(frameon=False, ncol=3) # Arrange legend in 3 columns
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"true_vs_pred_body{b}_time.png"), dpi=150)
            plt.close()
            
        print("Saved position-vs-time trajectory overlays to ./plots/")

        # error curve (RMSE over time)
        err = (traj_pred_cmp - traj_true_cmp).detach().cpu().numpy()
        rmse_t = np.sqrt((err**2).mean(axis=(1, 2)))
        plt.figure(figsize=(7, 4))
        plt.plot(rmse_t)
        plt.xlabel("time step"); plt.ylabel("RMSE")
        plt.title(f"Trajectory RMSE over time (first {T_cmp} steps)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "true_vs_pred_rmse_t.png"), dpi=150)
        plt.close()

        print("Saved true vs predicted trajectory overlays to ./plots/")

    if args.test_json and os.path.exists(args.test_json):
        if args.test_json.endswith(".npz"):
            print(f"Loading eval data from NPZ: {args.test_json}")
            data = np.load(args.test_json)
            # Create a 'jd' (json-data-like) dictionary to hold the tensors
            jd = {}
            # Load all arrays from NPZ into the dict
            for key in data:
                jd[key] = torch.from_numpy(data[key]).float()
            # Ensure 'dt' is a float, not a 0-dim tensor
            if 'dt' in jd:
                jd['dt'] = float(jd['dt'].item())
        else:
            # Fallback to original JSON loading
            print(f"Loading eval data from JSON: {args.test_json}")
            jd = load_json_data(args.test_json)

        # Try to get z0 from test JSON. If we detect a 3D X as traj_true, we'll reset z0 below.
        z0 = _try_get_tensor(jd, "z0", args.device, torch.float32)
        if z0 is None:
            X_any = _try_get_tensor(jd, "X", args.device, torch.float32)
            if X_any is not None:
                if X_any.ndim == 2:
                    z0 = X_any  # [B, D]
                elif X_any.ndim == 3 and X_any.shape[-1] == 6 * args.n_bodies:
                    # Treat X as a true trajectory; use its first frame as z0
                    z0 = X_any[0]  # [B, D]
                    jd["_X_was_traj_true"] = True  # mark for info message

        if z0 is not None:
            if "dt" in jd: args.dt = float(jd["dt"])

            # If we discovered that X is a trajectory, roll out to its length-1 for fair comparison.
            steps = args.rollout_steps
            if jd.get("_X_was_traj_true", False):
                steps = int(_try_get_tensor(jd, "X", args.device, torch.float32).shape[0]) - 1
            
            with torch.no_grad():
                traj = rollout(rk4_step, model.time_derivatives, z0, steps=steps, dt=args.dt)
            traj = traj.detach()

            drift = rel_energy_drift(model, traj)

            mean6 = drift[:6].mean(1).cpu().numpy()
            print(f"Eval(JSON): traj={tuple(traj.shape)}  mean relE first 6 steps={mean6}")

            outdir = _ensure_plots_dir("plots")

            # Diagnostic plots (pred only)
            q = _traj_to_q(traj, args.n_bodies)
            plot_energy_drift(drift, outdir)

            # Additional evaluation diagnostics
            # Compute energy, linear & angular momentum over time
            with torch.no_grad():
                Ht = torch.stack([model.hamiltonian(traj[t]) for t in range(traj.size(0))]).cpu().numpy()
                P  = torch.stack([total_linear_momentum(traj[t]) for t in range(traj.size(0))]).cpu().numpy()
                L  = torch.stack([angular_momentum(traj[t])       for t in range(traj.size(0))]).cpu().numpy()

            Ht_mean = Ht.mean(axis=1)
            P_mean  = P.mean(axis=1)
            L_mean  = L.mean(axis=1)

            # Energy vs time
            plt.figure(figsize=(7,4))
            plt.plot(Ht_mean, label="⟨H⟩")
            plt.xlabel("step"); plt.ylabel("Energy"); plt.title("Energy vs Time")
            plt.grid(alpha=0.3); plt.legend(frameon=False)
            plt.tight_layout(); plt.savefig(os.path.join(outdir, "energy_time.png"), dpi=150); plt.close()

            # Linear momentum components
            plt.figure(figsize=(7,4))
            for i, comp in enumerate(["x", "y", "z"]):
                plt.plot(P_mean[:, i], label=f"P_{comp}")
            plt.xlabel("step"); plt.ylabel("Momentum"); plt.title("Linear Momentum vs Time")
            plt.grid(alpha=0.3); plt.legend(frameon=False)
            plt.tight_layout(); plt.savefig(os.path.join(outdir, "linear_momentum_time.png"), dpi=150); plt.close()

            # Angular momentum components
            plt.figure(figsize=(7,4))
            for i, comp in enumerate(["x", "y", "z"]):
                plt.plot(L_mean[:, i], label=f"L_{comp}")
            plt.xlabel("step"); plt.ylabel("Angular Momentum"); plt.title("Angular Momentum vs Time")
            plt.grid(alpha=0.3); plt.legend(frameon=False)
            plt.tight_layout(); plt.savefig(os.path.join(outdir, "angular_momentum_time.png"), dpi=150); plt.close()

            # Per-body RMSE(t) if ground-truth trajectory available
            traj_true = jd.get("traj_true") or jd.get("traj") or jd.get("X")
            if traj_true is not None:
                if not torch.is_tensor(traj_true):
                    traj_true = torch.as_tensor(traj_true, device=traj.device, dtype=traj.dtype)

                # If 2-D [T, D], treat as single-trajectory batch [T, 1, D]
                if traj_true.ndim == 2:
                    traj_true = traj_true.unsqueeze(1)
                elif traj_true.ndim != 3:
                    raise SystemExit(f"Expected traj_true with ndim 2 or 3, got {traj_true.ndim}")

                T_cmp = min(traj.shape[0], traj_true.shape[0])
                ndof = 3 * args.n_bodies

                # Use ellipsis so both [T,B,D] and [T,1,D] work
                q_pred = traj[:T_cmp, ..., :ndof].reshape(T_cmp, -1, args.n_bodies, 3).cpu().numpy()
                q_true = traj_true[:T_cmp, ..., :ndof].reshape(T_cmp, -1, args.n_bodies, 3).cpu().numpy()

                bidx = 0
                for b in range(args.n_bodies):
                    diff = q_pred[:, bidx, b] - q_true[:, bidx, b]
                    rmse_t = np.sqrt((diff**2).mean(axis=1))
                    plt.figure(figsize=(6,4))
                    plt.plot(rmse_t)
                    plt.xlabel("step"); plt.ylabel(f"RMSE body {b}")
                    plt.title(f"Body {b} RMSE over time")
                    plt.tight_layout()
                    plt.savefig(os.path.join(outdir, f"rmse_time_body{b}.png"), dpi=150)
                    plt.close()

            with torch.no_grad():
                if hasattr(model, "log_m_body"):
                    m = model.log_m_body.exp().cpu().numpy()
                    plt.figure(figsize=(5,3))
                    plt.bar(range(len(m)), m)
                    plt.title("Learned Mass per Body")
                    plt.xlabel("Body index"); plt.ylabel("Mass")
                    plt.tight_layout()
                    plt.savefig(os.path.join(outdir, "learned_masses.png"), dpi=150); plt.close()
                    print("Learned masses:", np.round(m, 4))

            traj_rk4 = rollout(rk4_step, model.time_derivatives, z0, steps=args.rollout_steps, dt=args.dt)
            traj_lf  = rollout(leapfrog_step, model.time_derivatives, z0, steps=args.rollout_steps, dt=args.dt)

            Ht_rk4 = torch.stack([
                model.hamiltonian(traj_rk4[t]).detach()
                for t in range(traj_rk4.size(0))
            ]).cpu().numpy().mean(1)

            Ht_lf = torch.stack([
                model.hamiltonian(traj_lf[t]).detach()
                for t in range(traj_lf.size(0))
            ]).cpu().numpy().mean(1)

            plt.figure(figsize=(7,4))
            plt.plot(Ht_rk4, label="RK4")
            plt.plot(Ht_lf, label="Leapfrog")
            plt.xlabel("step"); plt.ylabel("Energy")
            plt.title("Energy Stability: RK4 vs Leapfrog")
            plt.legend(frameon=False)
            plt.tight_layout(); plt.savefig(os.path.join(outdir, "energy_rk4_vs_leapfrog.png"), dpi=150); plt.close()

            # try to overlay with ground truth if available or if X looked like [T,B,D]
            _overlay_true_pred_plots(jd, traj, args.n_bodies, outdir, args.device, traj.dtype)

            print(f"Saved plots to: {outdir}")
            test_used = True

    if not test_used and args.val_z0 is not None and os.path.exists(args.val_z0):
        z0 = load_npy(args.val_z0).to(args.device)
        traj = rollout(rk4_step, model.time_derivatives, z0, steps=args.rollout_steps, dt=args.dt)
        drift = rel_energy_drift(model, traj)

        mean6 = drift[:6].mean(1).cpu().numpy()
        print(f"Eval: traj={tuple(traj.shape)}  mean relE first 6 steps={mean6}")

        outdir = _ensure_plots_dir("plots")
        q = _traj_to_q(traj, args.n_bodies)
        plot_energy_drift(drift, outdir)

        # Additional evaluation diagnostics
        # Compute energy, linear & angular momentum over time
        with torch.no_grad():
            Ht = torch.stack([model.hamiltonian(traj[t]) for t in range(traj.size(0))]).cpu().numpy()
            P  = torch.stack([total_linear_momentum(traj[t]) for t in range(traj.size(0))]).cpu().numpy()
            L  = torch.stack([angular_momentum(traj[t])       for t in range(traj.size(0))]).cpu().numpy()

        Ht_mean = Ht.mean(axis=1)
        P_mean  = P.mean(axis=1)
        L_mean  = L.mean(axis=1)

        # Energy vs time
        plt.figure(figsize=(7,4))
        plt.plot(Ht_mean, label="⟨H⟩")
        plt.xlabel("step"); plt.ylabel("Energy"); plt.title("Energy vs Time")
        plt.grid(alpha=0.3); plt.legend(frameon=False)
        plt.tight_layout(); plt.savefig(os.path.join(outdir, "energy_time.png"), dpi=150); plt.close()

        # Linear momentum components
        plt.figure(figsize=(7,4))
        for i, comp in enumerate(["x", "y", "z"]):
            plt.plot(P_mean[:, i], label=f"P_{comp}")
        plt.xlabel("step"); plt.ylabel("Momentum"); plt.title("Linear Momentum vs Time")
        plt.grid(alpha=0.3); plt.legend(frameon=False)
        plt.tight_layout(); plt.savefig(os.path.join(outdir, "linear_momentum_time.png"), dpi=150); plt.close()

        # Angular momentum components
        plt.figure(figsize=(7,4))
        for i, comp in enumerate(["x", "y", "z"]):
            plt.plot(L_mean[:, i], label=f"L_{comp}")
        plt.xlabel("step"); plt.ylabel("Angular Momentum"); plt.title("Angular Momentum vs Time")
        plt.grid(alpha=0.3); plt.legend(frameon=False)
        plt.tight_layout(); plt.savefig(os.path.join(outdir, "angular_momentum_time.png"), dpi=150); plt.close()

        # Per-body RMSE(t) if ground-truth trajectory available
        traj_true_raw = jd.get("traj_true") or jd.get("traj") or jd.get("X")
        if traj_true_raw is not None:
            traj_true = torch.as_tensor(traj_true_raw, device=traj.device, dtype=traj.dtype)

            # Accept [T, D] or [T, B, D]. If it's 2-D, treat as single-trajectory batch.
            if traj_true.ndim == 2:           # [T, D] -> [T, 1, D]
                traj_true = traj_true.unsqueeze(1)
            elif traj_true.ndim != 3:
                raise SystemExit(f"Expected traj_true with ndim 2 or 3, got {traj_true.ndim}")

            # Sanity: last dim must match model’s D
            if traj_true.shape[-1] != traj.shape[-1]:
                raise SystemExit(f"traj_true D={traj_true.shape[-1]} != model D={traj.shape[-1]}")

            # Align + detach
            T_cmp = min(traj.shape[0], traj_true.shape[0])
            traj_pred_cmp = traj[:T_cmp].detach()
            traj_true_cmp = traj_true[:T_cmp].detach()

            # Extract positions [T, B, n_bodies, 3] using ellipsis (works for both cases)
            ndof = 3 * args.n_bodies
            q_pred = traj_pred_cmp[..., :ndof].reshape(T_cmp, -1, args.n_bodies, 3).cpu().numpy()
            q_true = traj_true_cmp[..., :ndof].reshape(T_cmp, -1, args.n_bodies, 3).cpu().numpy()

            bidx = 0  # or use args.plot_batch_index if needed
            for b in range(args.n_bodies):
                diff = q_pred[:, bidx, b] - q_true[:, bidx, b]  # [T_cmp, 3]
                rmse_t = np.sqrt((diff**2).mean(axis=1))
                plt.figure(figsize=(6,4))
                plt.plot(rmse_t)
                plt.xlabel("step"); plt.ylabel(f"RMSE body {b}")
                plt.title(f"Body {b} RMSE over time")
                plt.tight_layout()
                plt.savefig(os.path.join(outdir, f"rmse_time_body{b}.png"), dpi=150)
                plt.close()
        
        else:
            print("No ground-truth multi-step trajectory; skipping overlays.")

        with torch.no_grad():
            if hasattr(model, "log_m_body"):
                m = model.log_m_body.exp().cpu().numpy()
                plt.figure(figsize=(5,3))
                plt.bar(range(len(m)), m)
                plt.title("Learned Mass per Body")
                plt.xlabel("Body index"); plt.ylabel("Mass")
                plt.tight_layout()
                plt.savefig(os.path.join(outdir, "learned_masses.png"), dpi=150); plt.close()
                print("Learned masses:", np.round(m, 4))

        traj_rk4 = rollout(rk4_step, model.time_derivatives, z0, steps=args.rollout_steps, dt=args.dt)
        traj_lf  = rollout(leapfrog_step, model.time_derivatives, z0, steps=args.rollout_steps, dt=args.dt)

        Ht_rk4 = torch.stack([
            model.hamiltonian(traj_rk4[t]).detach()
            for t in range(traj_rk4.size(0))
        ]).cpu().numpy().mean(1)

        Ht_lf = torch.stack([
            model.hamiltonian(traj_lf[t]).detach()
            for t in range(traj_lf.size(0))
        ]).cpu().numpy().mean(1)

        plt.figure(figsize=(7,4))
        plt.plot(Ht_rk4, label="RK4")
        plt.plot(Ht_lf, label="Leapfrog")
        plt.xlabel("step"); plt.ylabel("Energy")
        plt.title("Energy Stability: RK4 vs Leapfrog")
        plt.legend(frameon=False)
        plt.tight_layout(); plt.savefig(os.path.join(outdir, "energy_rk4_vs_leapfrog.png"), dpi=150); plt.close()

        print(f"Saved plots to: {outdir}")

if __name__ == "__main__":
    main()