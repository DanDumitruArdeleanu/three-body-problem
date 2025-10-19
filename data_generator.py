import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import time
import json
import os

class DataGenerator():
    def __init__(self, D, n_bodies, max_steps, capture_steps, n, timestep,
                 start_dist = 0.01, eps = 0.1, G = 1, m = 1, box_halfwidth =0.1):
        self.D = D
        self.n_bodies = n_bodies
        self.max_steps = max_steps
        self.capture_steps = capture_steps
        self.n = n
        self.timestep = timestep
        self.start_dist = start_dist
        self.eps = eps
        self.G = G
        self.box_halfwidth = float(box_halfwidth)

        if np.isscalar(m):
            self.masses = np.full((n_bodies,), float(m))
        else:
            self.masses = np.asarray(m, dtype=float)
            assert self.masses.shape == (n_bodies,), \
                f"m must be scalar or shape ({n_bodies},), got {self.masses.shape}"
            assert np.all(self.masses > 0), "All masses must be > 0."
        
        self.X = []
        self.y = []

    def euclidian_distance(self, A, B):
        return np.linalg.norm(A - B)

    def sample_positions(self):
        D = self.D
        bodies = []

        # Working copies we can gently relax if placement gets hard
        hw = float(self.box_halfwidth)      # half-width of the placement box
        min_sep = float(self.start_dist)

        # Tunables for robustness (adjust if you like)
        max_tries_per_body = 5000           # hard cap per body
        expand_every = 1000                 # every N failed tries -> expand box
        relax_every  = 2000                 # every N failed tries -> relax min_sep

        for b in range(self.n_bodies):
            placed = False
            tries = 0
            while not placed:
                pos = np.random.uniform(-hw, hw, size=D)
                vel = np.random.uniform(-1, 1, size=D)

                if not bodies:
                    bodies.append(np.concatenate([pos, vel]))
                    placed = True
                    break

                # nearest distance to already-placed bodies
                dmin = min(self.euclidian_distance(pos, body[:D]) for body in bodies)

                if dmin >= min_sep:
                    bodies.append(np.concatenate([pos, vel]))
                    placed = True
                    break

                # Failed attempt -> update counters and (occasionally) relax constraints
                tries += 1
                if tries % expand_every == 0:
                    hw *= 1.1          # gently expand search domain
                if tries % relax_every == 0:
                    min_sep *= 0.95     # gently relax min separation

                if tries >= max_tries_per_body:
                    # Final fallback: place anyway and rely on softening (eps) to avoid blow-ups
                    bodies.append(np.concatenate([pos, vel]))
                    placed = True

        # Mass-aware COM velocity removal (unchanged)
        vels = np.stack([b[D:] for b in bodies])         # [N, D]
        masses = self.masses[:, None]                    # [N, 1]
        v_com = (masses * vels).sum(axis=0) / masses.sum()
        for i in range(self.n_bodies):
            bodies[i][D:] -= v_com

        return bodies


    def compute_accelerations(self, bodies):
        D = self.D
        accelerations = np.zeros((self.n_bodies, D))
        for i in range(self.n_bodies):
            for j in range(self.n_bodies):
                if i == j:
                    continue
                r_vec = bodies[j][:D] - bodies[i][:D]
                dist2 = np.dot(r_vec, r_vec) + self.eps**2
                inv_r3 = dist2 ** -1.5
                m_j = self.masses[j]          # source mass
                accelerations[i] += self.G * m_j * r_vec * inv_r3
        return accelerations


    def generate_training_data(self):
        D = self.D
        sims = 0
        
        while len(self.X) < self.n:
            # each body has [x, y, vx, vy] for D = 2
            # body[:D] = position | body[D:] = velocity
            bodies = self.sample_positions()
            
            state = np.concatenate(bodies)
            initial_state = state
            next_capture = 0
            accelerations = self.compute_accelerations(bodies)
            
            for t in range(1, self.max_steps+1):
                for i in range(self.n_bodies):
                    # update positions
                    # p(t + dt) = p(t) + v(t) * dt + 1/2 * a(t) * dt^2
                    bodies[i][:D] = bodies[i][:D] + bodies[i][D:] * self.timestep + 0.5 * accelerations[i] * (self.timestep**2)
                
                # calculate acceleration of each body
                new_accelerations = self.compute_accelerations(bodies)
                
                for i in range(self.n_bodies):
                    # update velocities
                    # v(t + dt) = v(t) + 1/2 * (a(t) + a(t+dt)) * dt
                    bodies[i][D:] = bodies[i][D:] + 0.5 * self.timestep * (accelerations[i] + new_accelerations[i])

                next_state = np.concatenate(bodies)
                
                if not self.capture_steps:
                    self.X.append(np.concatenate([state]))
                    self.y.append(next_state)
                    
                else:
                    if t == self.capture_steps[next_capture]:
                    
                        self.X.append(state)          # no extra scalar
                        self.y.append(next_state)
                        next_capture += 1
                        if next_capture >= len(self.capture_steps):
                            break

                    
                current_n = len(self.X)
    
                if current_n >= self.n:
                    break
                elif current_n % 1000 == 0:
                    print(f"{round((current_n/self.n)*100, 2)}% done -- {sims} simulations ran")
                
                accelerations = new_accelerations
                state = next_state
                
            sims += 1
        
        return np.array(self.X), np.array(self.y)
    
def reformat_batch_for_hnn(batch, masses, D, n_bodies):
    N = batch.shape[0]
    # [N, n_bodies, 2*D] -> (q,v) per body
    batch_reshaped = batch.reshape(N, n_bodies, 2 * D)
    
    # Extract all q's and v's
    q_batch = batch_reshaped[:, :, :D].reshape(N, n_bodies * D) # [N, n_bodies*D]
    v_batch = batch_reshaped[:, :, D:].reshape(N, n_bodies * D) # [N, n_bodies*D]
    
    # (p = m*v)
    # [n_bodies*D] = [m0,m0,m0, m1,m1,m1, ...]
    masses_tiled = np.repeat(masses, D)
    # v_batch * masses_tiled
    p_batch = v_batch * masses_tiled[None, :] 
    
    # Concatenate to [q_all, p_all]
    return np.concatenate([q_batch, p_batch], axis=1)
    
def main():
    start_time = time.time()
    capture_steps = [1, 2, 3, 5, 10]

    # set capture steps to None for sequential data
    train_generator = DataGenerator(D=3, n_bodies=3, max_steps=1000, capture_steps=None, 
                                    n=800, timestep=0.001)

    train_data = train_generator.generate_training_data()

    end_time = time.time()
    duration = end_time - start_time
    print(f"Finished in {round(duration, 3)} sec")

    test_generator = DataGenerator(D=3, n_bodies=3, max_steps=1000, capture_steps=None,
                                n=200, timestep=0.001)

    test_data = test_generator.generate_training_data()


    # =-=-=-=-=-=-=-=-= SAVE DATA =-=-=-=-=-=-=-=-=-=

    D_gen = train_generator.D
    n_b_gen = train_generator.n_bodies
    m_gen = train_generator.masses # [n_bodies]

    X_train, y_train = train_data
    
    # Convert to [q, p]
    X_train_hnn = reformat_batch_for_hnn(X_train, m_gen, D_gen, n_b_gen)
    y_train_hnn = reformat_batch_for_hnn(y_train, m_gen, D_gen, n_b_gen)

    train_dict = {
        "X": X_train_hnn.tolist(), # 'X', 'y'
        "y": y_train_hnn.tolist(),
        "dt": train_generator.timestep
    }

    X_test, y_test = test_data
    
    # Convert to HNN format
    X_test_hnn = reformat_batch_for_hnn(X_test, m_gen, D_gen, n_b_gen)
    y_test_hnn = reformat_batch_for_hnn(y_test, m_gen, D_gen, n_b_gen)

    test_dict = {
        "X": X_test_hnn.tolist(), # z0
        "y": y_test_hnn.tolist(), # z1
        "dt": test_generator.timestep
    }

    with open(os.path.join(os.getcwd(),"HNN_train.json"), "w") as f:
        json.dump(train_dict, f)
        
    with open(os.path.join(os.getcwd(),"HNN_test.json"), "w") as f:
        json.dump(test_dict, f)
        
    plt.plot(y_train[:1000])
    plt.show()

    print("PLOTTING TEST")

    plt.plot(X_test[:1000])
    plt.show()
    
if __name__ == "__main__":
    main()