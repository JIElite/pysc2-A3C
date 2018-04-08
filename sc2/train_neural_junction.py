import time

import numpy as np
import torch
from torch.autograd import Variable

from pysc2.lib import actions
from pysc2.lib import features

from envs import GameInterfaceHandler
from envs import create_pysc2_env
from model import FullyConv
from optimizer import ensure_shared_grad
from utils import freeze_layers

# action id
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_RIGHT_CLICK = actions.FUNCTIONS.Smart_screen.id
select_army = [actions.FunctionCall(_SELECT_ARMY, [[0]])]

# action parameters
_NOT_QUEUED = []

# Feature id
_SCREEN_SELECTED = features.SCREEN_FEATURES.selected.index
_SCREEN_PLAYER_ID = features.SCREEN_FEATURES.player_id.index
_SCREEN_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index


def train_simple_junction(worker_id, args, shared_model, optimizer, global_counter, summary_queue):
    torch.manual_seed(args['seed'] + worker_id)
    env_args = {
        'map_name': args['map'],
        'agent_race': args['agent_race'],
        'bot_race': args['bot_race'],
        'step_mul': args['step_mul'],
        'screen_size_px': [args['screen_resolution']] * 2,
        'minimap_size_px': [args['minimap_resolution']] * 2,
        'visualize': args['visualize'],
    }
    env = create_pysc2_env(env_args)
    game_inferface = GameInterfaceHandler(screen_resolution=args['screen_resolution'], minimap_resolution=args['minimap_resolution'])
    with env:
        local_model = FullyConv(screen_channels=8, screen_resolution=[args['screen_resolution']]*2).cuda()
        freeze_layers(local_model.conv1)
        freeze_layers(local_model.spatial_policy)
        freeze_layers(local_model.non_spatial_branch)
        freeze_layers(local_model.value)

        env.reset()
        state = env.step([actions.FunctionCall(_NO_OP, [])])[0]
        episode_done = False
        episode_length = 0
        episode_reward = 0
        local_step = 0

        while True:
            if local_step > args['worker_steps']:
                return

            # Sync the parameters with shared model
            local_model.load_state_dict(shared_model.state_dict())

            # Reset n-step experience buffer
            entropies = []
            critic_values = []
            spatial_policy_log_probs = []
            rewards = []


            # step forward n steps
            for step in range(args['n_steps']):
                local_step += 1
                screen_observation = Variable(torch.from_numpy(game_inferface.get_screen_obs(
                                                                    timesteps=state,
                                                                    indexes=[4, 5, 6, 7, 8, 9, 14, 15],
                                                                ))).cuda()
                spatial_action_prob, value = local_model(screen_observation)
                log_spatial_action_prob = torch.log(spatial_action_prob)
                spatial_action = spatial_action_prob.multinomial()
                spatial_entropy = -(log_spatial_action_prob * spatial_action_prob).sum(1)
                selected_log_action_prob = log_spatial_action_prob.gather(1, spatial_action)

                # record n-step experience
                entropies.append(spatial_entropy)
                spatial_policy_log_probs.append(selected_log_action_prob)
                critic_values.append(value)

                # Step
                action = game_inferface.build_action(_MOVE_SCREEN, spatial_action[0].cpu())
                state = env.step([action])[0]

                reward = np.asscalar(state.reward)
                rewards.append(reward)
                episode_reward += reward

                episode_length += 1
                global_counter.value += 1


                episode_done = (episode_length >= args['max_eps_length']) or state.last()
                if episode_done:
                    episode_length = 0
                    env.reset()
                    state = env.step([actions.FunctionCall(_NO_OP, [])])[0]
                    break

            R_t = torch.zeros(1)
            if not episode_done:
                screen_observation = Variable(torch.from_numpy(game_inferface.get_screen_obs(
                                                                    timesteps=state,
                                                                    indexes=[4, 5, 6, 7, 8, 9, 14, 15]
                                                            ))).cuda()
                _, value = local_model(screen_observation)
                R_t = value.data


            R_var = Variable(R_t).cuda()
            critic_values.append(R_var)
            policy_loss = 0.
            value_loss = 0.
            gae_ts = torch.zeros(1).cuda()
            for i in reversed(range(len(rewards))):
                R_var = rewards[i] + args['gamma'] * R_var

                advantage_var = R_var - critic_values[i]
                value_loss += 0.5 * advantage_var.pow(2)

                td_error = rewards[i] + args['gamma'] * critic_values[i+1].data - critic_values[i].data
                gae_ts = gae_ts * args['gamma'] * args['tau'] + td_error
                policy_loss += -(spatial_policy_log_probs[i] * Variable(gae_ts, requires_grad=False) + 0.05 *entropies[i])

            optimizer.zero_grad()
            total_loss = policy_loss + 0.5 * value_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm(local_model.parameters(), 40)
            ensure_shared_grad(local_model.conv2, shared_model.conv2)
            optimizer.step()

            if episode_done:
                summary_queue.put((global_counter.value, episode_reward))
                episode_reward = 0