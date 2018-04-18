import time

import numpy as np
import torch
from torch.autograd import Variable
from torch.distributions import Categorical

from pysc2.lib import actions
from pysc2.lib import features

from envs import GameInterfaceHandler
from envs import create_pysc2_env
from model import FullyConv, FullyConvMultiUnitCollectBaseline, FullyConvMultiUnitCollectBaselineExtended
from optimizer import ensure_shared_grad, ensure_shared_grad_cpu


torch.set_printoptions(threshold=5000)

# action id
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_RIGHT_CLICK = actions.FUNCTIONS.Smart_screen.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id

# action parameters
_NOT_QUEUED = []

# Feature id
_SCREEN_SELECTED = features.SCREEN_FEATURES.selected.index
_SCREEN_PLAYER_ID = features.SCREEN_FEATURES.player_id.index
_SCREEN_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index


def train_master(worker_id, args, shared_model, optimizer, global_counter, summary_queue):
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
        shared_master_model = shared_model[0]
        local_sub_model = shared_model[1]
        local_master_model = FullyConv(screen_channels=8, screen_resolution=[args['screen_resolution']]*2).cuda(args['gpu'])

        env.reset()
        state = env.step([actions.FunctionCall(_NO_OP, [])])[0]
        episode_done = False
        episode_length = 0
        episode_reward = 0

        while True:
            # Sync the parameters with shared model
            local_master_model.load_state_dict(shared_master_model.state_dict())

            # Reset n-step experience buffer
            entropies = []
            critic_values = []
            spatial_policy_log_probs = []
            rewards = []


            # step forward n steps
            for step in range(args['n_steps']):
                screen_observation = Variable(torch.from_numpy(game_inferface.get_screen_obs(
                                                                    timesteps=state,
                                                                    indexes=[4, 5, 6, 7, 8, 9, 14, 15],
                                                                ))).cuda(args['gpu'])

                select_spatial_action_prob, value = local_master_model(screen_observation)

                # mask spatial action
                selection_mask = torch.from_numpy((state.observation['screen'][_SCREEN_PLAYER_RELATIVE] == 1).astype('float32'))
                selection_mask = Variable(selection_mask.view(1, -1), requires_grad=False).cuda(args['gpu'])
                masked_select_spatial_action_prob = select_spatial_action_prob * selection_mask
                masked_select_spatial_action_prob = masked_select_spatial_action_prob / masked_select_spatial_action_prob.sum()
                # select_action = masked_select_spatial_action_prob.multinomial()
                m = Categorical(masked_select_spatial_action_prob)
                select_action = m.sample().unsqueeze(0)

                log_select_spatial_action_prob = torch.log(torch.clamp(masked_select_spatial_action_prob, min=1e-12))
                select_entropy = - (log_select_spatial_action_prob * masked_select_spatial_action_prob).sum(1)
                master_log_action_prob = log_select_spatial_action_prob.gather(1, select_action)
                # record n-step experience
                entropies.append(select_entropy)
                spatial_policy_log_probs.append(master_log_action_prob)
                critic_values.append(value)

                # Step
                action = game_inferface.build_action(_SELECT_POINT, select_action[0].cpu())
                state = env.step([action])[0]
                temp_reward = np.asscalar(state.reward)

                # sub agent step
                if _MOVE_SCREEN in state.observation['available_actions']:
                    # sub model decision
                    screen_observation = Variable(torch.from_numpy(game_inferface.get_screen_obs(
                        timesteps=state,
                        indexes=[4, 5, 6, 7, 8, 9, 14, 15],
                    )), volatile=True).cuda(args['gpu'])

                    spatial_action_prob, value = local_sub_model(screen_observation)
                    spatial_action = spatial_action_prob.multinomial()
                    action = game_inferface.build_action(_MOVE_SCREEN, spatial_action[0].cpu())
                    state = env.step([action])[0]
                else:
                    action = actions.FunctionCall(_NO_OP, [])
                    state = env.step([action])[0]

                temp_reward += np.asscalar(state.reward)

                # update episodic information
                rewards.append(temp_reward)
                episode_reward += temp_reward
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
                                                            ))).cuda(args['gpu'])
                _, value = local_master_model(screen_observation)
                R_t = value.data


            R_var = Variable(R_t).cuda(args['gpu'])
            critic_values.append(R_var)
            policy_loss = 0.
            value_loss = 0.
            gae_ts = torch.zeros(1).cuda(args['gpu'])
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
            torch.nn.utils.clip_grad_norm(local_master_model.parameters(), 40)
            ensure_shared_grad(local_master_model, shared_master_model)
            optimizer.step()

            if episode_done:
                summary_queue.put((global_counter.value, episode_reward))
                episode_reward = 0


def train_hierarchical_collect_baseline(
        worker_id, args, shared_model, optimizer, global_counter, summary_queue):

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
        game_inferface = GameInterfaceHandler(screen_resolution=args['screen_resolution'],
                                              minimap_resolution=args['minimap_resolution'])

        if args['extend_model']:
            model = FullyConvMultiUnitCollectBaselineExtended
        else:
            model = FullyConvMultiUnitCollectBaseline

        gpu_id = args['gpu']
        with env:
            local_model = model(
                screen_channels=8, screen_resolution=[args['screen_resolution']] * 2).cuda(gpu_id)


            env.reset()
            state = env.step([actions.FunctionCall(_NO_OP, [])])[0]
            episode_done = False
            episode_length = 0
            episode_reward = 0

            while True:

                # Sync the parameters with shared model
                local_model.load_state_dict(shared_model.state_dict())

                # Reset n-step experience buffer
                entropies = []
                critic_values = []
                chosen_select_policy_log_probs = []
                chosen_spatial_policy_log_probs = []
                rewards = []

                # step forward n steps
                for step in range(args['n_steps']):
                    screen_observation = Variable(torch.from_numpy(game_inferface.get_screen_obs(
                        timesteps=state,
                        indexes=[4, 5, 6, 7, 8, 9, 14, 15],
                    ))).cuda(gpu_id)

                    select_action_prob, spatial_action_prob, value = local_model(screen_observation)

                    # mask select action
                    selection_mask = torch.from_numpy(
                        (state.observation['screen'][_SCREEN_PLAYER_RELATIVE] == 1).astype('float32'))
                    selection_mask = Variable(selection_mask.view(1, -1), requires_grad=False).cuda(gpu_id)

                    masked_select_action_prob = select_action_prob * selection_mask
                    masked_select_action_prob = masked_select_action_prob / masked_select_action_prob.sum()
                    select_unit_action = masked_select_action_prob.multinomial()
                    select_action = game_inferface.build_action(_SELECT_POINT, select_unit_action[0].cpu())

                    spatial_action = spatial_action_prob.multinomial()
                    move_action = game_inferface.build_action(_MOVE_SCREEN, spatial_action[0].cpu())

                    log_select_action_prob = torch.log(torch.clamp(masked_select_action_prob, min=1e-15))
                    log_spatial_action_prob = torch.log(spatial_action_prob)

                    # compute entropy?
                    select_entropy = - (log_select_action_prob * masked_select_action_prob).sum(1)
                    spatial_entropy = - (log_spatial_action_prob * spatial_action_prob).sum(1)
                    entropy = select_entropy + spatial_entropy

                    chosen_log_select_unit_prob = log_select_action_prob.gather(1, select_unit_action)
                    chosen_log_spatial_action_prob = log_spatial_action_prob.gather(1, spatial_action)

                    # Step
                    state = env.step([select_action])[0]
                    reward = np.asscalar(state.reward)
                    state = env.step([move_action])[0]
                    reward += np.asscalar(state.reward)

                    # record n-step data
                    entropies.append(entropy)
                    chosen_select_policy_log_probs.append(chosen_log_select_unit_prob)
                    chosen_spatial_policy_log_probs.append(chosen_log_spatial_action_prob)
                    rewards.append(reward)
                    critic_values.append(value)

                    # update statistical information
                    episode_reward += reward
                    episode_length += 1
                    global_counter.value += 1

                    episode_done = (episode_length >= args['max_eps_length']) or state.last()
                    if episode_done:
                        episode_length = 0
                        env.reset()
                        state = env.step([actions.FunctionCall(_NO_OP, [])])[0]
                        break

                # TODO update model, compute gradient
                R_t = torch.zeros(1)
                if not episode_done:
                    screen_observation = Variable(torch.from_numpy(game_inferface.get_screen_obs(
                        timesteps=state,
                        indexes=[4, 5, 6, 7, 8, 9, 14, 15]
                    ))).cuda(gpu_id)
                    _, _, value = local_model(screen_observation)
                    R_t = value.data

                R_var = Variable(R_t).cuda(gpu_id)
                critic_values.append(R_var)
                policy_loss = 0.
                value_loss = 0.
                gae_ts = torch.zeros(1).cuda(gpu_id)
                for i in reversed(range(len(rewards))):
                    R_var = rewards[i] + args['gamma'] * R_var

                    advantage_var = R_var - critic_values[i]
                    value_loss += 0.5 * advantage_var.pow(2)

                    td_error = rewards[i] + args['gamma'] * critic_values[i + 1].data - critic_values[i].data
                    gae_ts = gae_ts * args['gamma'] * args['tau'] + td_error

                    policy_log_for_action = chosen_select_policy_log_probs[i] + chosen_spatial_policy_log_probs[i]
                    policy_loss += -(policy_log_for_action * Variable(gae_ts, requires_grad=False) + 0.05 * entropies[i])

                optimizer.zero_grad()
                total_loss = policy_loss + 0.5 * value_loss
                total_loss.backward()
                torch.nn.utils.clip_grad_norm(local_model.parameters(), 40)

                if args['multiple_gpu']:
                    ensure_shared_grad_cpu(local_model, shared_model)
                else:
                    ensure_shared_grad(local_model, shared_model)

                optimizer.step()


                if episode_done:
                    summary_queue.put((global_counter.value, episode_reward))
                    episode_reward = 0



def train_conjunction_collect(
    worker_id, args, shared_model, optimizer, global_counter, summary_queue):
    pass