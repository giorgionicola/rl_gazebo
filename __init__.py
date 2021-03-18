import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)


### HUMAN ROBOT ###

register(
    id='HumanCylRobotEnv-v0',
    entry_point='rl_gazebo_env.envs.human_robot:HumanCylUrEnv0'
)

