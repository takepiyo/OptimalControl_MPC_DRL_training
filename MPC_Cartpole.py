import numpy as np
import cvxopt
from cvxopt import matrix
from scipy import linalg
import matplotlib.pyplot as plt
import gym


def opt_mpc_with_input_const(A, B, N, Q, R, P, x0, umax=None, umin=None):
    (nx, nu) = B.shape

    Ai = A
    AA = Ai
    for i in range(2, N + 1):
        Ai = A.dot(Ai)
        AA = np.vstack((AA, Ai))

    AiB = B
    BB = np.kron(np.eye(N), AiB)
    for i in range(1, N):
        AiB = A.dot(AiB)
        BB += np.kron(np.diag(np.ones(N - i), -i), AiB)

    RR = np.kron(np.eye(N), R)
    QQ = linalg.block_diag(np.kron(np.eye(N - 1), Q), P)

    H = (BB.T.dot(QQ).dot(BB) + RR)

    gx0 = BB.T.dot(QQ).dot(AA).dot(x0)
    P = matrix(H)
    q = matrix(gx0)

    if umax is None and umin is None:
        sol = cvxopt.solvers.qp(P, q)
    else:
        G = np.zeros((0, nu * N))
        h = np.zeros((0, 1))

        if umax is not None:
            tG = np.eye(N * nu)
            th = np.kron(np.ones((N * nu, 1)), umax)
            G = np.vstack([G, tG])
            h = np.vstack([h, th])

        if umin is not None:
            tG = np.eye(N * nu) * -1.0
            th = np.kron(np.ones((N * nu, 1)), umin * -1.0)
            G = np.vstack([G, tG])
            h = np.vstack([h, th])

        G = matrix(G)
        h = matrix(h)

        sol = cvxopt.solvers.qp(P, q, G, h)

    u = np.matrix(sol["x"])
    xx = AA.dot(x0) + BB.dot(u)
    x = xx.reshape(N, nx)
    # x = np.concatenate([x0.transpose(), xx.reshape(N, nx)], axis=0)

    x = np.array(x)
    u = np.array(u)
    return x, u


def plot_log_and_predict(axes, x, u, obs_log, control_log, t, predict_horizon):
    obs_log = np.stack(obs_log)
    control_log = np.stack(control_log)
    axes[0].clear()
    axes[1].clear()
    axes[2].clear()
    axes[0].plot(obs_log[:, 0:2])
    axes[1].plot(obs_log[:, 2:4])
    axes[2].plot(control_log)
    predict_timestep = np.arange(start=t, stop=t + predict_horizon)
    axes[0].plot(predict_timestep, x[:, 0:2])
    axes[1].plot(predict_timestep, x[:, 2:4])
    axes[2].plot(predict_timestep, u)
    axes[0].legend(["x", "x_dot", "x_pre", "x_dot_pre"])
    axes[1].legend(["theta", "theta_dot", "theta_pre", "theta_dot_pre"])
    axes[2].set_title("force")
    axes[0].grid(True)
    axes[1].grid(True)
    axes[2].grid(True)
    plt.pause(1e-8)


def main():
    alpha = 10.0
    beta = 1.0
    max_step = 500
    predict_horizon = 100

    env = gym.make('CartPole-v0')

    gravity = env.gravity
    masscart = env.masscart
    masspole = env.masspole
    length = env.length
    dt = env.tau

    k = length * (4. / 3 - (masspole / (masspole + masscart)))

    A = np.array([[0., 1., 0., 0.],
                  [0., 0., gravity / k, 0.],
                  [0., 0., 0., 1.],
                  [0., 0., gravity / k, 0.]]) * dt + np.eye(4)

    B = np.array([[0.], [1. / (masscart + masspole)], [0.], [-1. / k]]) * dt
    (nx, nu) = B.shape

    Q = alpha * np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    R = beta * np.eye(nu)
    P = Q
    obs = env.reset()
    x0 = obs.reshape(-1, 1)

    control_log = []
    obs_log = []
    total_reward = 0

    fig, axes = plt.subplots(nx + nu - 2, tight_layout=True, figsize=(10, 50))

    for t in range(max_step):
        x, u = opt_mpc_with_input_const(A=A, B=B, N=predict_horizon, Q=Q, R=R,
                                        P=P, x0=x0, umax=10.0, umin=-10.0)
        force = u[0]
        env.env.force_mag = force[0]
        obs, reward, done, _ = env.step(1)
        x0 = obs.reshape(-1, 1)
        total_reward += reward
        env.render()
        control_log.append(force)
        obs_log.append(obs)
        plot_log_and_predict(axes, x, u, obs_log,
                             control_log, t, predict_horizon)
    print(f"{total_reward=}")


if __name__ == '__main__':
    main()
