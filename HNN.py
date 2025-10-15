import argparse, os, json
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import random
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Utilities
def seed_all(seed=0):
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

        # Normalization buffers
        self.register_buffer("z_mean", torch.zeros(2*self.ndof))
        self.register_buffer("z_std",  torch.ones(2*self.ndof))

        if separable:
            self.V_net = build_mlp(in_dim=self.ndof, out_dim=1, hidden=hidden, depth=depth)
            if learn_mass:
                if tie_body_mass:
                    self.log_m_body = nn.Parameter(torch.zeros(self.n_bodies))  # per body
                else:
                    self.log_m = nn.Parameter(torch.zeros(self.ndof))           # per DOF
            else:
                if tie_body_mass:
                    self.register_buffer("log_m_body", torch.zeros(self.n_bodies))
                else:
                    self.register_buffer("log_m", torch.zeros(self.ndof))
        else:
            self.mlp = build_mlp(in_dim=2 * self.ndof, out_dim=1, hidden=hidden, depth=depth)

        # Canonical symplectic form J = [[0, I], [-I, 0]]
        J = torch.zeros(2*self.ndof, 2*self.ndof)
        J[:self.ndof, self.ndof:] = torch.eye(self.ndof)
        J[self.ndof:, :self.ndof] = -torch.eye(self.ndof)
        self.register_buffer("J", J)

    # Energies
    def kinetic(self, p: torch.Tensor) -> torch.Tensor:
        # p: [B, ndof]
        if self.tie_body_mass:
            M = self.log_m_body.exp().repeat_interleave(3)  # [ndof]
        else:
            M = self.log_m.exp()
        Minv = 1.0 / (M + 1e-8)
        return 0.5 * (p.pow(2) * Minv).sum(dim=1, keepdim=True)  # [B,1]

    def potential(self, q: torch.Tensor) -> torch.Tensor:
        # Normalize q part only
        qn = (q - self.z_mean[:self.ndof]) / (self.z_std[:self.ndof] + 1e-8)
        return self.V_net(qn)  # [B,1]

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
            # don't sever the graph; just ensure z requires grad
            if not z.requires_grad:
                z = z.clone().requires_grad_(True)

            H = self.hamiltonian(z)                             # [B]
            # Keep the graph so gradients can flow through RK4 unrolled steps
            grad = torch.autograd.grad(H.sum(), z, create_graph=True)[0]  # [B, 2*ndof]
            dzdt = torch.matmul(grad, self.J.t())               # [B, 2*ndof]
            dqdt, dpdt = torch.split(dzdt, [self.ndof, self.ndof], dim=1)

            if self.constraint_fn is not None:
                dqdt, dpdt = self._apply_constraints(z, dqdt, dpdt)

            return torch.cat([dqdt, dpdt], dim=1)

    def _apply_constraints(self, z: torch.Tensor, dqdt: torch.Tensor, dpdt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Finzi et al. (CHNN): project velocities into tangent space and apply constraint forces.
        B = z.size(0)
        q = z[:, :self.ndof].detach().requires_grad_(True)
        g = self.constraint_fn(q.view(B, self.n_bodies, 3))  # [B,k]
        if g.ndim == 1:
            g = g.unsqueeze(1)
        k = g.size(1)

        # Build J = ∂g/∂q : [B,k,ndof]
        rows = []
        for i in range(k):
            gi = g[:, i].sum()
            Ji = torch.autograd.grad(gi, q, retain_graph=True, allow_unused=False)[0]  # [B, ndof]
            rows.append(Ji)
        J = torch.stack(rows, dim=1)  # [B,k,ndof]

        eye_k = torch.eye(k, device=z.device).unsqueeze(0)
        A = torch.bmm(J, J.transpose(1,2)) + 1e-6 * eye_k  # [B,k,k]

        # 1) Project velocities onto tangent space
        rhs_v = torch.bmm(J, dqdt.unsqueeze(-1)).squeeze(-1)  # [B,k]
        try:
            L = torch.linalg.cholesky(A)
            lam_v = torch.cholesky_solve(rhs_v.unsqueeze(-1), L).squeeze(-1)
        except RuntimeError:
            lam_v = torch.linalg.solve(A, rhs_v)
        dqdt = dqdt - torch.bmm(J.transpose(1,2), lam_v.unsqueeze(-1)).squeeze(-1)

        # 2) Constraint forces correct dpdt
        rhs_f = torch.bmm(J, dqdt.unsqueeze(-1)).squeeze(-1)
        try:
            lam_f = torch.cholesky_solve(rhs_f.unsqueeze(-1), L).squeeze(-1)
        except Exception:
            lam_f = torch.linalg.solve(A, rhs_f)
        dpdt = dpdt - torch.bmm(J.transpose(1,2), lam_f.unsqueeze(-1)).squeeze(-1)
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
def train_epoch_vfield(model: HNN, x: torch.Tensor, ydot: torch.Tensor, batch_size: int = 1024, lr: float = 1e-3) -> float:
    # Vector-field supervision (Greydanus): minimize ||f_theta(z) - \dot{z}||^2.
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    total = 0.0
    N = x.size(0)
    for i in range(0, N, batch_size):
        xb = x[i:i+batch_size]
        yb = ydot[i:i+batch_size]
        pred = model.time_derivatives(xb)
        loss = F.mse_loss(pred, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * xb.size(0)
    return total / N

def train_epoch_rollout(model: HNN, z_t: torch.Tensor, z_tp1: torch.Tensor, dt: float, batch_size: int = 1024, lr: float = 1e-3) -> float:
    # One-step RK4 rollout supervision (Neural-ODE flavor from HNN-ODEs repo).
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    total = 0.0
    N = z_t.size(0)
    for i in range(0, N, batch_size):
        zb = z_t[i:i+batch_size]
        zb_next = z_tp1[i:i+batch_size]
        z_pred = rk4_step(model.time_derivatives, zb, dt)
        loss = F.mse_loss(z_pred, zb_next)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * zb.size(0)
    return total / N

def train_epoch_rolloutK(model: HNN, z_t: torch.Tensor, z_targets: torch.Tensor, dt: float, K: int = 4, batch_size: int = 256, lr: float = 1e-3) -> float:
    # Multi-step free rollout supervision (short horizon), common in ODE-style training.
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
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
    parser.add_argument("--train-json", type=str, default="HNN_train.json", help="JSON with training data (z, dz or z_next, optional dt)")
    parser.add_argument("--test-json", type=str, default="HNN_test.json", help="JSON with eval data (z0 or z, optional dt)")
    parser.add_argument("--train-z", type=str, help="Path to training states z.npy [N, D]")
    parser.add_argument("--train-dz", type=str, default=None, help="Path to training vector field dz/dt.npy [N, D] (vfield mode)")
    parser.add_argument("--train-z-next", type=str, default=None, help="Path to next states z_{t+1}.npy [N, D] (rollout mode)")
    parser.add_argument("--val-z0", type=str, default=None, help="Path to eval initial states z0.npy [B, D]")
    parser.add_argument("--dt", type=float, default=1e-3, help="Timestep for RK4 rollout/labels")
    # Model
    parser.add_argument("--n-bodies", type=int, required=False, help="Auto-detected from z if not set")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--separable", action="store_true", help="Use separable H=T(p)+V(q) (default if flag set)")
    parser.add_argument("--no-learn-mass", action="store_true", help="Disable learned masses")
    parser.add_argument("--no-tie-body-mass", action="store_true", help="Do not tie masses across x/y/z per body")
    # Constraint
    parser.add_argument("--constraint", type=str, default=None, choices=[None, "anchor", "pair"])
    parser.add_argument("--anchor-radius", type=float, default=1.0)
    parser.add_argument("--pair-i", type=int, default=0)
    parser.add_argument("--pair-j", type=int, default=1)
    parser.add_argument("--pair-dist", type=float, default=1.0)
    # Train
    parser.add_argument("--mode", type=str, default="vfield", choices=["vfield", "rollout"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=420)
    # Eval
    parser.add_argument("--rollout-steps", type=int, default=200)
    # IO
    parser.add_argument("--save", type=str, default="./hnn.pt")
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    seed_all(args.seed)

    # Load training data (prefer JSON if available)
    z = dz = z_next = None  # initialize so they're always defined

    if args.train_json and os.path.exists(args.train_json):
        jd = load_json_data(args.train_json)
        if 'dt' in jd:
            args.dt = jd['dt']
        z = jd.get('z')
        dz = jd.get('dz')
        z_next = jd.get('z_next')

        if z is None:
            raise SystemExit("HNN_train.json must contain at least 'z'")

        # move tensors to device
        z = z.to(args.device)
        if dz is not None:
            dz = dz.to(args.device)
        if z_next is not None:
            z_next = z_next.to(args.device)

        print(f"Loaded training JSON: {args.train_json}")

        # If next-state labels are provided (and no dz), prefer rollout training
        if z_next is not None and dz is None and args.mode != "rollout":
            print("Detected next-state labels in training JSON -> switching to rollout training.")
            args.mode = "rollout"

    else:
        if args.train_z is None:
            raise SystemExit("--train-z is required when no train JSON is provided")
        z = load_npy(args.train_z).to(args.device)
        print(f"Loaded z: {tuple(z.shape)} from {args.train_z}")

    N, D = z.shape
    # Auto-detect n_bodies if not provided
    if args.n_bodies is None:
        if D % 6 != 0:
            raise SystemExit(f"Cannot infer n_bodies from z.shape={z.shape}; expected D divisible by 6")
        args.n_bodies = D // 6
        print(f"Auto-detected n_bodies = {args.n_bodies}")
    assert D == 2*3*args.n_bodies, "z dimensionality must match n_bodies"

    # Auto-enable separable if not explicitly set
    if not args.separable:
        print("Auto-enabling separable Hamiltonian H(q,p)=T(p)+V(q).")
        args.separable = True


    # Build model
    constraint_fn = build_constraint(args, args.n_bodies)
    model = HNN(
        n_bodies=args.n_bodies,
        hidden=args.hidden,
        depth=args.depth,
        constraint_fn=constraint_fn,
        separable=args.separable,
        learn_mass=not args.no_learn_mass,
        tie_body_mass=not args.no_tie_body_mass,
    ).to(args.device)

    # Normalization from training states
    mean, std = fit_norm(z)
    apply_norm(model, mean.to(args.device), std.to(args.device))

    # Load checkpoint (optional)
    if args.load is not None and os.path.exists(args.load):
        print(f"Loading checkpoint: {args.load}")
        load_checkpoint(args.load, model, map_location=args.device)

    # Train
    if args.mode == "vfield":
        if dz is None:
            if args.train_dz is None:
                raise SystemExit("--train-dz is required in vfield mode (or provide dz in HNN_train.json)")
            dz = load_npy(args.train_dz).to(args.device)
        assert dz.shape == z.shape
        for ep in range(1, args.epochs+1):
            loss = train_epoch_vfield(model, z, dz, batch_size=args.batch, lr=args.lr)
            print(f"[vf] epoch {ep:03d} | loss {loss:.6f}")
    else:
        if z_next is None:
            if args.train_z_next is None:
                raise SystemExit("--train-z-next is required in rollout mode (or provide z_next in HNN_train.json)")
            z_next = load_npy(args.train_z_next).to(args.device)
        assert z_next.shape == z.shape
        for ep in range(1, args.epochs+1):
            loss = train_epoch_rollout(model, z, z_next, args.dt, batch_size=args.batch, lr=args.lr)
            print(f"[ro] epoch {ep:03d} | loss {loss:.6f}")

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

    def plot_xy_all_bodies(q, outdir, max_trajs=16):
        """
        q: [T, B, n_bodies, 3]
        Saves: one plot per body overlaying up to max_trajs trajectories (x-y).
        """
        T, B, n_bodies, _ = q.shape
        use_B = min(B, max_trajs)

        for b in range(n_bodies):
            plt.figure(figsize=(6, 6))
            for i in range(use_B):
                plt.plot(q[:, i, b, 0], q[:, i, b, 1], linewidth=0.8)
            plt.title(f"Trajectories (x–y) — Body {b} (first {use_B} trajs)")
            plt.xlabel("x"); plt.ylabel("y"); plt.axis("equal"); plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"traj_body{b}_xy_all.png"), dpi=150)
            plt.close()

    def plot_xy_per_traj(q, outdir, bodies="all", max_trajs=8):
        """
        q: [T, B, n_bodies, 3]
        Saves: per-trajectory plots with all bodies together (x–y).
        """
        T, B, n_bodies, _ = q.shape
        use_B = min(B, max_trajs)
        if bodies == "all":
            bodies_idx = range(n_bodies)
        else:
            bodies_idx = bodies

        for i in range(use_B):
            plt.figure(figsize=(6, 6))
            for b in bodies_idx:
                plt.plot(q[:, i, b, 0], q[:, i, b, 1], linewidth=0.9, label=f"body {b}")
            plt.title(f"Traj {i} — all bodies (x–y)")
            plt.xlabel("x"); plt.ylabel("y"); plt.axis("equal"); plt.legend(frameon=False)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"traj{i:02d}_allbodies_xy.png"), dpi=150)
            plt.close()

    def plot_3d_each_body(q, outdir, max_trajs=4):
        """
        q: [T, B, n_bodies, 3]
        Saves: per-body 3D plots (first max_trajs trajectories).
        """
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, import registers 3D
        T, B, n_bodies, _ = q.shape
        use_B = min(B, max_trajs)

        for b in range(n_bodies):
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection="3d")
            for i in range(use_B):
                ax.plot(q[:, i, b, 0], q[:, i, b, 1], q[:, i, b, 2], linewidth=0.8)
            ax.set_title(f"Body {b} — 3D trajectories (first {use_B})")
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"traj_body{b}_3d.png"), dpi=150)
            plt.close()

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
        traj_pred_cmp = traj_pred[:T_cmp]
        traj_true_cmp = traj_true[:T_cmp]

        # Extract positions as [T, B, n_bodies, 3]
        def _to_q(t):  # [T,B,D] with D=6*n_bodies, q then p
            q = t[..., :3 * n_bodies]
            return q.reshape(T_cmp, -1, n_bodies, 3)

        q_pred = _to_q(traj_pred_cmp).detach().cpu().numpy()
        q_true = _to_q(traj_true_cmp).detach().cpu().numpy()

        # Pick a single trajectory index for clarity (first in batch)
        bidx = 0
        Path(outdir).mkdir(parents=True, exist_ok=True)

        # Per-body overlays
        for b in range(n_bodies):
            plt.figure(figsize=(6, 6))
            plt.plot(q_true[:, bidx, b, 0], q_true[:, bidx, b, 1], '-',  alpha=0.85, label='true')
            plt.plot(q_pred[:, bidx, b, 0], q_pred[:, bidx, b, 1], '--', alpha=0.9,  label='pred')
            plt.title(f"Body {b} — True vs Predicted (x–y)")
            plt.xlabel("x"); plt.ylabel("y"); plt.axis("equal")
            plt.grid(True, alpha=0.3); plt.legend(frameon=False)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"true_vs_pred_body{b}_xy.png"), dpi=150)
            plt.close()

        # All bodies together
        plt.figure(figsize=(6, 6))
        for b in range(n_bodies):
            plt.plot(q_true[:, bidx, b, 0], q_true[:, bidx, b, 1], '-',  alpha=0.65, label=f"true b{b}")
            plt.plot(q_pred[:, bidx, b, 0], q_pred[:, bidx, b, 1], '--', alpha=0.85, label=f"pred b{b}")
        plt.title("All bodies — True vs Predicted (x–y)")
        plt.xlabel("x"); plt.ylabel("y"); plt.axis("equal")
        plt.grid(True, alpha=0.3); plt.legend(frameon=False, ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "true_vs_pred_allbodies_xy.png"), dpi=150)
        plt.close()

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

            traj = rollout(rk4_step, model.time_derivatives, z0, steps=steps, dt=args.dt)
            drift = rel_energy_drift(model, traj)

            mean6 = drift[:6].mean(1).cpu().numpy()
            print(f"Eval(JSON): traj={tuple(traj.shape)}  mean relE first 6 steps={mean6}")

            outdir = _ensure_plots_dir("plots")

            # Diagnostic plots (pred only)
            q = _traj_to_q(traj, args.n_bodies)
            plot_xy_all_bodies(q, outdir, max_trajs=16)
            plot_xy_per_traj(q, outdir, bodies="all", max_trajs=8)
            plot_3d_each_body(q, outdir, max_trajs=4)
            plot_energy_drift(drift, outdir)

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
        plot_xy_all_bodies(q, outdir, max_trajs=16)
        plot_xy_per_traj(q, outdir, bodies="all", max_trajs=8)
        plot_3d_each_body(q, outdir, max_trajs=4)
        plot_energy_drift(drift, outdir)
        print(f"Saved plots to: {outdir}")

if __name__ == "__main__":
    main()
