from itertools import count
import sys

from absl import app, flags
from pysc2.env import sc2_env
from pysc2.lib import actions

import numpy as np
import torch
from torch.autograd import Variable

from envs import create_pysc2_env, GameInterfaceHandler
from model import FullyConv


FLAGS = flags.FLAGS
# Game related settings
flags.DEFINE_string("map", "CollectMineralShardsSingle", "Name of a map to use.")
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

# action type id
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_RIGHT_CLICK = actions.FUNCTIONS.Smart_screen.id


def main(argv):
    torch.manual_seed(FLAGS.seed)

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
    game_inferface = GameInterfaceHandler(screen_resolution=FLAGS.screen_resolution,
                                          minimap_resolution=FLAGS.minimap_resolution)
    action_id = _NO_OP
    if FLAGS.map == 'CollectMineralShardsSingle':
        action_id = _MOVE_SCREEN
    elif FLAGS.map == 'DefeatBuilding3':
        action_id = _RIGHT_CLICK

    # agent's model
    model = FullyConv(screen_channels=8,
                             screen_resolution=(FLAGS.screen_resolution, FLAGS.screen_resolution)).cuda()
    # load model
    model_destination = './models/backup_model_best'
    # model_destination = './models/task1_15625916/model_best'
    model.load_state_dict(torch.load(model_destination))

    # play with environment
    with env:
        for i_episode in range(FLAGS.num_episodes):
            env.reset()
            state = env.step([actions.FunctionCall(_NO_OP, [])])[0]
            episodic_reward = 0

            for step in count():
                screen_observation = Variable(torch.from_numpy(game_inferface.get_screen_obs(
                    timesteps=state,
                    indexes=[4, 5, 6, 7, 8, 9, 14, 15],
                ))).cuda()
                spatial_action_prob, log_spatial_action_prob, value = model(screen_observation)
                spatial_action = spatial_action_prob.multinomial()

                # Step
                action = game_inferface.build_action(action_id, spatial_action[0].cpu())
                state = env.step([action])[0]

                reward = np.asscalar(state.reward)
                episodic_reward += reward

                episode_done = (step >= FLAGS.max_eps_length) or state.last()
                if episode_done:
                    print('Episodic reward:', episodic_reward)
                    break


if __name__ == '__main__':
    app.run(main)
