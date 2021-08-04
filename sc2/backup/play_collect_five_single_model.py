"""
command example:
python play_collect_five_single_model.py --gpu 1 --model_type 3 \
 --pretrained_model models/collect_five_nj_extend_conv3_18134363/model_best --visualize True


we can use input() to control each step
"""

from itertools import count
import sys
import time

from absl import app, flags
from pysc2.env import sc2_env
from pysc2.lib import actions, features

import numpy as np
import seaborn as sns
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

from envs import create_pysc2_env, GameInterfaceHandler
from model import (
    FullyConvMultiUnitCollectBaseline,
    FullyConvMultiUnitCollectBaselineExtended,
    Grafting_MultiunitCollect,
    ExtendConv3Grafting_MultiunitCollect,
)


FLAGS = flags.FLAGS
# Game related settings
flags.DEFINE_string("map", "CollectMineralShardsFiveExtended", "Name of a map to use.")
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
flags.DEFINE_integer("model_type", 0, "model type for single model/grafted model")
flags.DEFINE_integer("gpu", 0, "gpu device")
flags.DEFINE_string("pretrained_model", "", "path of pretrained model")
flags.DEFINE_float("step_delay", 0.3, "delay for each agent step")
FLAGS(sys.argv)

torch.cuda.set_device(FLAGS.gpu)
print("CUDA device:", torch.cuda.current_device())


# action type id
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id

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
    env = create_pysc2_env(env_args)
    game_inferface = GameInterfaceHandler(screen_resolution=FLAGS.screen_resolution,
                                          minimap_resolution=FLAGS.minimap_resolution)

    model_type = FLAGS.model_type
    if model_type == 0:
        model = FullyConvMultiUnitCollectBaseline
    elif model_type == 1:
        model = FullyConvMultiUnitCollectBaselineExtended
    elif model_type == 2:
        model = Grafting_MultiunitCollect
    elif model_type == 3:
        model = ExtendConv3Grafting_MultiunitCollect
    else:
        print("Error: Wrong model!")
        return

    with env:
        local_model = model(screen_channels=8, screen_resolution=[FLAGS.screen_resolution] * 2).cuda()
        local_model.load_state_dict(torch.load(FLAGS.pretrained_model))


        total_eps_return = 0
        for i_episode in range(FLAGS.num_episodes):
            env.reset()
            state = env.step([actions.FunctionCall(_NO_OP, [])])[0]
            episodic_reward = 0

            for step in count():
                screen_observation = Variable(torch.from_numpy(game_inferface.get_screen_obs(
                    timesteps=state,
                    indexes=[4, 5, 6, 7, 8, 9, 14, 15],
                )), volatile=True).cuda()
                select_action_prob, spatial_action_prob, value, spatial_2d = local_model(screen_observation)

                # mask select action
                selection_mask = torch.from_numpy(
                    (state.observation['screen'][_SCREEN_PLAYER_RELATIVE] == 1).astype('float32'))
                selection_mask = Variable(selection_mask.view(1, -1), requires_grad=False).cuda()
                selection_mask = torch.clamp(selection_mask, min=1e-8)

                masked_select_action_prob = select_action_prob * selection_mask
                masked_select_action_prob = masked_select_action_prob / masked_select_action_prob.sum()

                select_unit_action = masked_select_action_prob.multinomial()
                select_action = game_inferface.build_action(_SELECT_POINT, select_unit_action[0].cpu())

                # Step
                state = env.step([select_action])[0]
                reward = np.asscalar(state.reward)

                # visualize spatial action prob
                # if step % 5 == 0:
                #     np_spatial = spatial_2d.data.cpu().numpy()
                #     sns.heatmap(np_spatial, linewidths=0.5, cmap="YlGnBu")
                #     plt.show()
                #     input()

                if _MOVE_SCREEN in state.observation['available_actions']:
                    # spatial_action = spatial_action_prob.multinomial()
                    # move_action = game_inferface.build_action(_MOVE_SCREEN, spatial_action[0].cpu())

                    spatial_action = spatial_action_prob.max(1)[1]
                    move_action = game_inferface.build_action(_MOVE_SCREEN, spatial_action.cpu())
                    state = env.step([move_action])[0]

                else:
                    state = env.step([actions.FunctionCall(_NO_OP, [])])[0]

                time.sleep(FLAGS.step_delay)

                reward += np.asscalar(state.reward)
                episodic_reward += reward

                episode_done = (step >= FLAGS.max_eps_length) or state.last()
                if episode_done:
                    print('Episodic reward:', episodic_reward)
                    total_eps_return += episodic_reward
                    break

        print("mean episode return:", total_eps_return / FLAGS.num_episodes)

if __name__ == '__main__':
    app.run(main)