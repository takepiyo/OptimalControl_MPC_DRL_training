import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math


class IndependentTwoWheeledRobot(gym.Env):
    metadata = {"render.modes": [
        "human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self):
        self.wheel_radius = 1.0
        self.wheel_distance = 15.0

        self.tau = 0.02  # seconds between state updates

        self.x_threshold = 100.0
        self.y_threshold = 100.0

        high = np.array([self.x_threshold, np.finfo(np.float32).max, self.y_threshold, np.finfo(
            np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max])
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        high = np.array([10.0, 10.0])
        self.action_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        x, x_dot, y, y_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        B = np.array([[0.5 * self.wheel_radius * costheta, 0.5 * self.wheel_radius * costheta], [0., 0.], [0.5 * self.wheel_radius *
                     sintheta, 0.5 * self.wheel_radius * sintheta], [0., 0.], [0.5 / self.wheel_distance, - 0.5 / self.wheel_distance], [0., 0.]])

        self.state += B.dot(action) * self.tau

        done = False

        if not done:
            reward = 1.0

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.0, high=0.0, size=(6,))
        return np.array(self.state, dtype=np.float32)

    def render(self, mode="human"):
        screen_width = 800
        screen_height = 600

        world_width = self.x_threshold * 2
        world_height = self.y_threshold * 2
        scale_x = screen_width / world_width
        scale_y = screen_height / world_height
        robot_radius = self.wheel_distance

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            robot = rendering.Compound(
                [rendering.make_circle(robot_radius, filled=False), rendering.Line(start=(0, 0), end=(robot_radius, 0))])
            self.robottrans = rendering.Transform()
            robot.add_attr(self.robottrans)
            self.viewer.add_geom(robot)

        if self.state is None:
            return None

        x, x_dot, y, y_dot, theta, theta_dot = self.state
        self.robottrans.set_translation(
            x * scale_x + screen_width / 2.0, y * scale_y + screen_height / 2.0)
        self.robottrans.set_rotation(theta)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
