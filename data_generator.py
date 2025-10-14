import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import time
import json
import os

class DataGenerator():
    def __init__(self, D, n_bodies, max_steps, capture_steps, n, timestep,
                 start_dist = 0.01, eps = 0.1, G = 1, m = 1):
        self.D = D
        self.n_bodies = n_bodies
        self.max_steps = max_steps
        self.capture_steps = capture_steps
        self.n = n
        self.timestep = timestep
        self.start_dist = start_dist
        self.eps = eps
        self.G = G
        self.m = m
        
        self.X = []
        self.y = []

    def euclidian_distance(self, A, B):
        return np.linalg.norm(A - B)

    def sample_positions(self):
        D = self.D
        bodies = []

        for _ in range(self.n_bodies):
            pos = None
            vel = None
            smallest_distance = 0
            while smallest_distance < self.start_dist:
                pos = np.random.uniform(-0.5, 0.5, size=D)
                vel = np.random.uniform(-1, 1, size=D)
                smallest_distance = float('inf')
                for body in bodies:
                    dist = self.euclidian_distance(pos, body[:D])
                    if dist < smallest_distance:
                        smallest_distance = dist
                    
            bodies.append(np.concatenate([pos, vel]))

        # zero total momentum to prevent drift
        total_momentum = np.sum([b[D:] for b in bodies], axis=0)
        for b in bodies:
            b[D:] -= total_momentum / float(self.n_bodies)

        return bodies

    def compute_accelerations(self, bodies):
        D = self.D
        
        accelerations = np.zeros((self.n_bodies, D))
        for i in range(self.n_bodies):
            for j in range(self.n_bodies):
                if i != j:
                    r_vec = bodies[j][:D] - bodies[i][:D]
                    dist2 = np.dot(r_vec, r_vec) + self.eps**2
                    inv_dist3 = dist2 ** -1.5
                    # G and mass are 1 for simplicity
                    accelerations[i] += self.G * self.m * r_vec * inv_dist3
                            
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
                    
                        self.X.append(np.concatenate([initial_state, [self.timestep * t]]))
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

    X_train, y_train = train_data
    X_train = X_train.tolist()
    y_train = y_train.tolist()

    train_dict = {
        "X": X_train,
        "y": y_train 
    }

    X_test, y_test = test_data
    X_test = X_test.tolist()

    test_dict = {
        "X": X_test,
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