from itertools import count
import time
import sys
import os
os.environ['OMP_NUM_THREADS']='1'

from absl import app, flags
from pysc2.env import sc2_env
from pysc2.lib import actions, features

from envs import create_pysc2_env, GameInterfaceHandler
import numpy as np
import torch
from torch.autograd import Variable


from model2 import (
    CollectAndDestroyGraftingNet,
    CollectAndDestroyGraftingDropoutNet,
    CollectAndDestroyGraftingDropoutNetConv4,
    CollectAndDestroyGraftingDropoutNetConv6,
    CollectAndDestroyBaseline,
    CollectAndDestroyGraftingDropoutNetBN
)


FLAGS = flags.FLAGS
# Game related settings
flags.DEFINE_string("map", "CollectAndDestroyAirSCV", "Name of a map to use.")
flags.DEFINE_integer("screen_resolution", 48, "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 48, "Resolution for minimap feature layers.")
flags.DEFINE_bool("visualize", False, "Whether to render with pygame.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_enum("agent_race", None, sc2_env.races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", None, sc2_env.races.keys(), "Bot's race.")
flags.DEFINE_integer('max_eps_length', 5000, "max length run for each episode")

# Learning related settings
flags.DEFINE_float("gamma", 0.99, "Discount rate for future rewards.")
flags.DEFINE_integer("n_steps", 8,  "How many steps do we compute the Return (TD)")
flags.DEFINE_integer("num_episodes", 100, "# of episode for agent to play with environment")
flags.DEFINE_integer("seed", 5, "torch random seed")
flags.DEFINE_integer("gpu", 0, "gpu device")
flags.DEFINE_integer("version", 0, "version of network")
flags.DEFINE_integer('transfer', 0, 'transfer module type')
FLAGS(sys.argv)

# PySC2 actions
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_RIGHT_CLICK = actions.FUNCTIONS.Smart_screen.id

# PySC2 features
_SCREEN_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index


# torch.cuda.set_device(FLAGS.gpu)
torch.cuda.set_device(1)
print("CUDA device:", torch.cuda.current_device())


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


    # model
    if FLAGS.version == 0:
        model = CollectAndDestroyGraftingNet
    elif FLAGS.version == 1:
        model = CollectAndDestroyGraftingDropoutNet
    elif FLAGS.version == 2:
        model = CollectAndDestroyGraftingDropoutNetBN
    elif FLAGS.version == 3:
        model = CollectAndDestroyBaseline
    elif FLAGS.version == 4:
        model = CollectAndDestroyGraftingDropoutNetConv6


    print("model type:", model)

    agent = model(screen_channels=8, screen_resolution=(FLAGS.screen_resolution, FLAGS.screen_resolution)).cuda()
    # agent.load_state_dict(torch.load('./models/model_latest_test'))
    agent.load_state_dict(torch.load('./models/model_latest_collect_and_destroy_air_scv_conv6_bn_wo_annealing_7'))

    agent.train()
    agent.dropout_rate = 0.95
    # agent.eval()

    print('----  load model successfully. ----')

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
                ))).cuda()

                select_unit_action_prob, value, selected_task = agent(screen_observation,
                                                                    Variable(torch.zeros(1, 1, 48, 48).cuda()),
                                                                    0)

                # select unit
                selection_mask = torch.from_numpy(
                    (state.observation['screen'][_SCREEN_PLAYER_RELATIVE] == 1).astype('float32'))
                selection_mask = Variable(selection_mask.view(1, -1), requires_grad=False).cuda()
                masked_select_unit_action_prob = select_unit_action_prob * selection_mask

                if float(masked_select_unit_action_prob.sum().cpu().data.numpy()) < 1e-12:
                    masked_select_unit_action_prob += 1.0 * selection_mask
                    masked_select_unit_action_prob /= masked_select_unit_action_prob.sum()
                else:
                    masked_select_unit_action_prob = masked_select_unit_action_prob / masked_select_unit_action_prob.sum()

                try:
                    select_action = masked_select_unit_action_prob.multinomial()
                except:
                    print("Error detect!")
                    print(masked_select_unit_action_prob)


                # select task type
                task = selected_task.multinomial()
                action = game_inferface.build_action(_SELECT_POINT, select_action[0].cpu())
                state = env.step([action])[0]

                time.sleep(0.5)

                if state.reward > 1:
                    reward = np.asscalar(np.array([10]))
                else:
                    reward = np.asscalar(np.array([-0.2]))
                episodic_reward += reward

                episode_done = (step >= FLAGS.max_eps_length) or state.last()
                if episode_done:
                    env.reset()
                    state = env.step([actions.FunctionCall(_NO_OP, [])])[0]
                    break

                task = int(task.cpu().data.numpy())
                if task == 0 and _MOVE_SCREEN in state.observation['available_actions']:
                    # collection mineral shards
                    screen_observation = Variable(torch.from_numpy(game_inferface.get_screen_obs(
                        timesteps=state,
                        indexes=[4, 5, 6, 7, 8, 9, 14, 15],
                    ))).cuda()

                    spatial_action_prob, value, _ = agent(screen_observation,
                                                        Variable(torch.ones(1, 1, 48, 48)).cuda(), 1)
                    spatial_action = spatial_action_prob.multinomial()

                    action = game_inferface.build_action(_MOVE_SCREEN, spatial_action[0].cpu())
                    state = env.step([action])[0]
                    if state.reward > 1:
                        reward = np.asscalar(np.array([10]))
                    else:
                        reward = np.asscalar(np.array([-0.2]))

                elif task == 1 and _RIGHT_CLICK in state.observation['available_actions']:
                    # destroy enemy's buildings
                    screen_observation = Variable(torch.from_numpy(game_inferface.get_screen_obs(
                        timesteps=state,
                        indexes=[4, 5, 6, 7, 8, 9, 14, 15],
                    ))).cuda()

                    destroy_action_prob, value, _ = agent(screen_observation,
                                                                Variable(torch.ones(1, 1, 48, 48) * 2).cuda(), 2)
                    destroy_position = destroy_action_prob.multinomial()

                    action = game_inferface.build_action(_RIGHT_CLICK, destroy_position[0].cpu())
                    state = env.step([action])[0]
                    if state.reward > 1:
                        reward = np.asscalar(np.array([10]))
                    else:
                        reward = np.asscalar(np.array([-0.2]))

                else:
                    action = actions.FunctionCall(_NO_OP, [])
                    state = env.step([action])[0]

                    if state.reward > 1:
                        reward = np.asscalar(np.array([10]))
                    else:
                        reward = np.asscalar(np.array([-0.2]))

                time.sleep(0.5)

                episodic_reward += reward

                episode_done = (step >= FLAGS.max_eps_length) or state.last()
                if episode_done:
                    env.reset()
                    state = env.step([actions.FunctionCall(_NO_OP, [])])[0]
                    break

            print('eps reward:', episodic_reward)
            total_eps_reward += episodic_reward

    mean_performance = total_eps_reward / FLAGS.num_episodes
    print("Mean performance:", mean_performance)


if __name__ == '__main__':
    app.run(main)