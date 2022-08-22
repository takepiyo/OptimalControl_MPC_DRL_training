import numpy as np
import matplotlib.pyplot as plt
import math

from tqdm import tqdm


class DoubleSlit1DSOC:
    def __init__(self, tf, slit_t, dt, v, R, alpha, slit1, slit2, x_min, x_max) -> None:
        self.tf = tf
        self.slit_t = slit_t
        self.dt = dt
        self.v = v
        self.R = R
        self.alpha = alpha

        self.slit1 = slit1
        self.slit2 = slit2
        self.x_min = x_min
        self.x_max = x_max

        self.sigma = lambda t: math.sqrt(v * (tf - t))
        self.sigma_1 = lambda t: math.sqrt(self.sigma(
            t) ** 2 * v * R / (alpha * self.sigma(t) ** 2 + v * R))
        self.A = lambda t: 1 / (self.slit_t - t) + 1 / (R + tf - slit_t)
        self.B = lambda x, t: x / (slit_t - t)
        self.F = lambda x_0, x, t: math.erf(
            math.sqrt(self.A(t) / (2 * v)) * (x_0 - (self.B(x, t) / self.A(t))))
        self.P = lambda x, t: self.F(
            slit1[1], x, t) - self.F(slit1[0], x, t) + self.F(slit2[1], x, t) - self.F(slit2[0], x, t)
        self.J = lambda x, t: v * R * math.log(self.sigma(t) / self.sigma_1(t)) + 0.5 * (
            self.sigma_1(t) * x / self.sigma(t)) ** 2 - v * R * math.log(0.5 * (self.P(x, t)) + 1e-5) if t < slit_t else v * R * math.log(self.sigma(t) / self.sigma_1(t)) + 0.5 * (
            self.sigma_1(t) * x / self.sigma(t)) ** 2 * alpha
        self.partial_xF = lambda x_0, x, t: 2 / math.sqrt(math.pi) * math.exp(-math.sqrt(
            self.A(t) / (2 * v)) * (x_0 - self.B(x, t) / self.A(t)) ** 2)
        self.partial_xP = lambda x, t: self.partial_xF(slit1[1], x, t) - self.partial_xF(
            slit1[0], x, t) + self.partial_xF(slit2[1], x, t) - self.partial_xF(slit2[0], x, t)

        self.optimal_u = lambda x, t: - (v * x) / (R + tf - t) - (self.partial_xP(x, t) / self.P(x, t)) * (v / (
            math.sqrt(2 * v * self.A(t)) * (slit_t - t))) if t < slit_t else - (alpha * x) / (R + alpha * (tf - t))

    def command(self, x, t):
        return self.optimal_u(x, t)

    def draw_J(self):
        fig, ax = plt.subplots()
        x = np.arange(self.x_min, self.x_max, 0.01)
        for t in [0.0, self.slit_t - self.dt, self.slit_t + self.dt, self.tf - self.dt]:
            ax.plot(x, list(map(lambda x: self.J(x, t), x)),
                    label="t={}".format(t))
        ax.legend()
        plt.show()


class DoubleSlitPathIntegral:
    def __init__(self, env, N) -> None:
        self.env = env
        self.N = N
        self.dt = env.dt
        self.v = env.v
        self.phi = env.phi

        self.sampled_cost = []
        self.noise_history = []
        self.traj = []
        for _ in tqdm(range(N)):
            x, t, done = env.reset()
            while not done:
                x, t, done = env.step(u=0)
            if env.n + 1 == env.max_n:
                self.sampled_cost.append(math.exp(-self.phi(x) / self.v))
                self.noise_history.append(env.noise_array)
            self.traj.append((env.n, env.x_array))
        self.noise_history = np.stack(self.noise_history)
        # env.render_multiple_path(self.traj)

        self.psi = sum(self.sampled_cost) / N

    def command(self, x, t):
        return np.sum(self.sampled_cost * (self.noise_history[:, int(t / self.dt)])) / self.psi
        # return np.sum(self.sampled_cost * (self.noise_history[:, 1])) / self.psi


class DoubleSlit1D:
    def __init__(self) -> None:
        self.T = 2.0
        self.dt = 0.02
        self.slit_t = 1.0
        self.slit_n = self.slit_t / self.dt
        self.max_n = self.T / self.dt

        self.x_min = -10.0
        self.x_max = 10.0
        self.slit1 = [-6.0, -4.0]
        self.slit2 = [4.0, 6.0]
        self.V = lambda x, n: 10000 if n == self.slit_n and (not ((
            self.slit1[0] < x < self.slit1[1]) or (self.slit2[0] < x < self.slit2[1]))) else 0
        self.alpha = 1.0
        self.phi = lambda x: 0.5 * self.alpha * x ** 2
        self.v = 1.0
        self.R = 0.1

    def reset(self):
        self.x = 0.0
        self.wiener = 0.0
        self.t = 0.0
        self.n = 0

        self.t_array = np.arange(0, self.T, self.dt)
        self.x_array = np.zeros_like(self.t_array)
        self.x_array[self.n] = self.x
        self.noise_array = np.zeros_like(self.t_array)

        done = True if self.n == self.max_n else False
        return self.x, self.t_array[self.n], done

    def step(self, u):
        self.n += 1
        noise = np.sqrt(self.dt) * np.random.randn()
        self.noise_array[self.n] = noise
        self.x += u * self.dt + noise
        self.x_array[self.n] = self.x
        done = (self.n + 1 == self.max_n or self.V(self.x, self.n) > 0 or not (
            self.x_min < self.x < self.x_max))
        return self.x, self.t_array[self.n], done

    def render(self):
        figure, ax = plt.subplots()
        ax.set_ylim(self.x_min, self.x_max)
        ax.set_xlim(0, self.T)
        ax.set_aspect(0.08)
        ax.plot([self.slit_t, self.slit_t],
                [self.x_min, self.slit1[0]], color="black")
        ax.plot([self.slit_t, self.slit_t], [
            self.slit1[1], self.slit2[0]], color="black")
        ax.plot([self.slit_t, self.slit_t],
                [self.slit2[1], self.x_max], color="black")

        ax.plot(self.t_array[: self.n + 1], self.x_array[: self.n + 1])
        plt.show()

    def render_multiple_path(self, x_arrays):
        fig, ax = plt.subplots()
        ax.set_ylim(self.x_min, self.x_max)
        ax.set_xlim(0, self.T)
        ax.set_aspect(0.08)
        ax.plot([self.slit_t, self.slit_t],
                [self.x_min, self.slit1[0]], color="black")
        ax.plot([self.slit_t, self.slit_t], [
            self.slit1[1], self.slit2[0]], color="black")
        ax.plot([self.slit_t, self.slit_t],
                [self.slit2[1], self.x_max], color="black")

        for n, x_array in x_arrays:
            ax.plot(self.t_array[: n + 1], x_array[: n + 1])
        plt.show()


if __name__ == "__main__":
    env = DoubleSlit1D()
    ctrl = DoubleSlit1DSOC(env.T, env.slit_t, env.dt, env.v, env.R,
                           env.alpha, env.slit1, env.slit2, env.x_min, env.x_max)
    # ctrl = DoubleSlitPathIntegral(env, 100000)
    traj = []
    for _ in tqdm(range(100)):
        x, t, done = env.reset()
        while not done:
            u = ctrl.command(x, t)
            x, t, done = env.step(u=u)
        traj.append((env.n, env.x_array))
    env.render_multiple_path(traj)
    # ctrl.draw_J()
