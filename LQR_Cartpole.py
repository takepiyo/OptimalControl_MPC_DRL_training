import gym
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg


def solve_riccati_iter(A, B, Q, R, tolerance=1e-5, max_iter=1e6):
    pass


def solve_riccati_arimoto_potter(A, B, Q, R):
    pass


def main():
    alpha = 5.0
    beta = 1.0

    env = gym.make('CartPole-v0')
    env.seed(1)

    gravity = env.gravity
    masscart = env.masscart
    masspole = env.masspole
    length = env.length

    k = length * (4. / 3 - (masspole / (masspole + masscart)))

    A = np.array([[0., 1., 0., 0.],
                  [0., 0., gravity / k, 0.],
                  [0., 0., 0., 1.],
                  [0., 0., gravity / k, 0.]])

    B = np.array([[0.], [1. / (masscart + masspole)], [0.], [-1. / k]])

    Q = alpha * np.eye(4)
    R = beta * np.eye(1)

    P_iter = None
    P_potter = None
    P_scipy = linalg.solve_continuous_are(A, B, Q, R)

    K_scipy = np.linalg.inv(R).dot(B.T.dot(P_scipy))

    obs = env.reset()
    obs_log = [obs]
    control_log = []
    total_reward = 0

    for t in range(500):
        force = -K_scipy.dot(obs)[0]
        control_log.append(force)
        env.env.force_mag = force
        obs, reward, done, _ = env.step(1)
        obs_log.append(obs)
        total_reward += reward
        env.render()
        if done:
            print(f"{t=},{total_reward=}")
            fig, ax = plt.subplots(2)
            ax[0].plot(obs_log)
            ax[0].legend(['x', 'x_dot', 'theta', 'theta_dot'])
            ax[1].plot(control_log)
            plt.show()


if __name__ == "__main__":
    main()
