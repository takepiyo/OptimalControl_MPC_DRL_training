from importlib.metadata import entry_points
from gym.envs.registration import register

register(
    id='IndependentTwoWheeledRobot-v0',
    entry_point='Env.Independent_TwoWheeled_Robot:IndependentTwoWheeledRobot'
)
