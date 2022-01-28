import gym
import Env
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('IndependentTwoWheeledRobot-v0')

obs = env.reset()

obses = []

for i in range(300):
    env.render()
    # obs, reward, done, info = env.step(env.action_space.sample())
    obs, reward, done, info = env.step(
        np.array([-5.0, 5.0], dtype=np.float32))
    obses.append(obs)
plt.plot(obses)
plt.legend(["x", "x_dot", "y", "y_dot", "theta", "theta_dot"])
plt.show()
env.close()
