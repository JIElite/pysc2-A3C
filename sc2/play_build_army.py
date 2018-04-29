import sys
from itertools import count
import time

import numpy as np
from pysc2.lib import actions
from pysc2.env import sc2_env
from absl import flags, app

from envs import create_pysc2_env


_NO_OP = actions.FUNCTIONS.no_op.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id

_NOT_QUEUED = [0]
_SELECT_ALL = [0]


class BuildArmyAgent:
    def __init__(self):
        pass

    def step(self, obs):
        available_actions = obs.observation['available_actions']


        if _TRAIN_MARINE in available_actions:
            print("action:", _TRAIN_MARINE)
            return actions.FunctionCall(_TRAIN_MARINE, [_NOT_QUEUED])
        else:
            print("action:", _NO_OP)
            return actions.FunctionCall(_NO_OP, [])




FLAGS = flags.FLAGS
# Game related settings
flags.DEFINE_string("map", "BuildArmy", "Name of a map to use.")
flags.DEFINE_integer("screen_resolution", 32, "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 32, "Resolution for minimap feature layers.")
flags.DEFINE_bool("visualize", False, "Whether to render with pygame.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_enum("agent_race", None, sc2_env.races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", None, sc2_env.races.keys(), "Bot's race.")
flags.DEFINE_integer('total_steps', 10500000, "steps run for each worker")
flags.DEFINE_integer('max_eps_length', 5000, "max length run for each episode")

# Learning related settings
flags.DEFINE_integer("seed", 5, "torch random seed")
flags.DEFINE_integer("num_episodes", 100, "# of episode for agent to play with environment")
flags.DEFINE_string("model_desination", "", "destination of pretrained model")
FLAGS(sys.argv)


def main(argv):
    # build environment
    env_args = {
        'map_name': FLAGS.map,
        'agent_race': FLAGS.agent_race,
        'bot_race': FLAGS.bot_race,
        'step_mul': FLAGS.step_mul,
        'screen_size_px': [FLAGS.screen_resolution] * 2,
        'minimap_size_px': [FLAGS.minimap_resolution] * 2,
        'visualize': FLAGS.visualize,
    }
    env = create_pysc2_env(env_args)
    # play with environment

    agent = BuildArmyAgent()

    with env:
        for i_episode in range(FLAGS.num_episodes):
            env.reset()
            state = env.step([actions.FunctionCall(_NO_OP, [])])[0]
            episodic_reward = 0

            for step in count():
                action = agent.step(state)
                state = env.step([action])[0]
                time.sleep(0.1)

                reward = np.asscalar(state.reward)
                episodic_reward += reward

                episode_done = (step >= FLAGS.max_eps_length) or state.last()
                if episode_done:
                    print('Episodic reward:', episodic_reward)
                    break


if __name__ == '__main__':
    app.run(main)



if __name__ == '__main__':
    main()
