from itertools import count
import sys

from absl import app, flags
from pysc2.env import sc2_env
from pysc2.lib import actions, features

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.autograd import Variable


from envs import create_pysc2_env, GameInterfaceHandler
from model import (
    FullyConv,
    FullyConvExtended,
)


FLAGS = flags.FLAGS
# Game related settings
flags.DEFINE_string("map", "CollectMineralShardsSingleExtended", "Name of a map to use.")
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
flags.DEFINE_integer("gpu", 0, "gpu device")
flags.DEFINE_integer("num_episodes", 100, "# of episode for agent to play with environment")
flags.DEFINE_string("model_desination", "", "destination of pretrained model")
flags.DEFINE_boolean("extend_model", False, "using extended model or not")
flags.DEFINE_float("step_delay", 0.3, "delay for each agent step")
FLAGS(sys.argv)

torch.cuda.set_device(FLAGS.gpu)
print("CUDA device:", torch.cuda.current_device())

# action type id
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id

# features id
_SCREEN_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index


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
    gpu_id = FLAGS.gpu
    env = create_pysc2_env(env_args)
    game_inferface = GameInterfaceHandler(screen_resolution=FLAGS.screen_resolution,
                                          minimap_resolution=FLAGS.minimap_resolution)

    if FLAGS.extend_model:
        model = FullyConvExtended
    else:
        model = FullyConv

    long_term_model = FullyConv(screen_channels=8, screen_resolution=[FLAGS.screen_resolution] * 2).cuda(gpu_id)
    long_term_model.load_state_dict(torch.load('./models/task1_insert_no_op_steps_6_3070000/model_latest_insert_no_op_steps_6'))

    short_term_model = FullyConv(screen_channels=8, screen_resolution=[FLAGS.screen_resolution] * 2).cuda(gpu_id)
    short_term_model.load_state_dict(torch.load('./models/task1_300s_original_16347668/model_latest'))

    # play with environment
    total_eps_reward = 0
    with env:
        for i_episode in range(FLAGS.num_episodes):
            env.reset()
            state = env.step([actions.FunctionCall(_NO_OP, [])])[0]
            episodic_reward = 0

            for step in count(start=1):
                screen_observation = Variable(torch.from_numpy(game_inferface.get_screen_obs(
                    timesteps=state,
                    indexes=[4, 5, 6, 7, 8, 9, 14, 15],
                ))).cuda(gpu_id)

                spatial_action_prob, value, long_term_spatial_policy = long_term_model(screen_observation)
                spatial_action = spatial_action_prob.multinomial()
                move_action = game_inferface.build_action(_MOVE_SCREEN, spatial_action[0].cpu())

                # Step
                state = env.step([move_action])[0]
                reward = np.asscalar(state.reward)
                episodic_reward += reward


                plt.figure()
                plt.subplot(121)
                plt.title('long term policy')
                np_long_term = long_term_spatial_policy.data.cpu().numpy()
                sns.heatmap(np_long_term, linewidths=0.5, cmap='YlGnBu')

                plt.subplot(122)
                plt.title('short term policy')
                _, _, short_term_spatial_policy = short_term_model(screen_observation)
                np_short_term = short_term_spatial_policy.data.cpu().numpy()
                sns.heatmap(np_short_term, linewidths=0.5, cmap='YlGnBu')
                plt.show()

                # Action delay
                # time.sleep(FLAGS.step_delay)

                episode_done = (step >= FLAGS.max_eps_length) or state.last()
                if episode_done:
                    print('Episodic reward:', episodic_reward)
                    total_eps_reward += episodic_reward
                    break

    mean_performance = total_eps_reward / FLAGS.num_episodes
    print("Mean performance:", mean_performance)


if __name__ == '__main__':
    app.run(main)