import time

import numpy as np
import torch
from torch.autograd import Variable
from torch.distributions import Categorical

from pysc2.lib import actions
from pysc2.lib import features

from envs import GameInterfaceHandler
from envs import create_pysc2_env
from model import (
    Grafting_MultiunitCollect,
    ExtendConv3Grafting_MultiunitCollect,
    Grafting_MultiunitCollect_WithActionFeatures,
    ExtendConv3Grafting_MultiunitCollect_WithActionFeatures,
    MultiInputSinglePolicyNet,
    MultiInputSinglePolicyNetExtendConv3,
)
from model2 import CollectFiveDropout, CollectFiveDropoutConv3
from optimizer import ensure_shared_grad, ensure_shared_grad_cpu
from utils import freeze_layers

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




def train_conjunction_with_action_features(
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

    if args['version'] == 1:
        if args['extend_model']:
            model = ExtendConv3Grafting_MultiunitCollect
        else:
            model = Grafting_MultiunitCollect
    elif args['version'] == 2:
        if args['extend_model']:
            model = ExtendConv3Grafting_MultiunitCollect_WithActionFeatures
        else:
            model = Grafting_MultiunitCollect_WithActionFeatures
    elif args['version'] == 3:
        if args['extend_model']:
            model = MultiInputSinglePolicyNetExtendConv3
        else:
            model = MultiInputSinglePolicyNet
    elif args['version'] == 4:
        if args['extend_model']:
            model = CollectFiveDropoutConv3
        else:
            model = CollectFiveDropout

    gpu_id = args['gpu']
    with env:
        local_model = model(screen_channels=8, screen_resolution=[args['screen_resolution']] * 2).cuda(gpu_id)
        local_model.train()
        # freeze_layers(shared_model.conv_master)
        # freeze_layers(shared_model.conv_sub)
        # freeze_layers(shared_model.spatial_policy)
        # freeze_layers(shared_model.select_unit)

        env.reset()
        state = env.step([actions.FunctionCall(_NO_OP, [])])[0]
        episode_done = False
        episode_length = 0
        episode_reward = 0

        while True:
            # Sync the parameters with shared model
            local_model.load_state_dict(shared_model.state_dict())
            if args['version'] == 4:
                local_model.anneal_dropout()

            # Reset n-step experience buffer
            entropies = []
            critic_values = []
            chosen_log_policy_probs = []
            rewards = []

            # step forward n steps
            for step in range(args['n_steps']):
                screen_observation = Variable(torch.from_numpy(game_inferface.get_screen_obs(
                    timesteps=state,
                    indexes=[4, 5, 6, 7, 8, 9, 14, 15],
                ))).cuda(gpu_id)

                select_indicator = Variable(torch.zeros(1, 1, 32, 32)).cuda(gpu_id)

                if args['version'] == 2:
                    select_action_prob, _, value, _ = local_model(screen_observation, select_indicator)
                elif args['version'] == 4:
                    select_action_prob, _, value, _ = local_model(screen_observation, select_indicator, 0)
                else:
                    select_action_prob, value, _ = local_model(screen_observation, select_indicator)

                # mask select action
                selection_mask = torch.from_numpy(
                    (state.observation['screen'][_SCREEN_PLAYER_RELATIVE] == 1).astype('float32'))
                selection_mask = Variable(selection_mask.view(1, -1), requires_grad=False).cuda(gpu_id)
                selection_mask = torch.clamp(selection_mask, min=1e-8)

                masked_select_action_prob = select_action_prob * selection_mask
                masked_select_action_prob = masked_select_action_prob / masked_select_action_prob.sum()

                select_unit_action = masked_select_action_prob.multinomial()
                select_action = game_inferface.build_action(_SELECT_POINT, select_unit_action[0].cpu())
                log_select_action_prob = torch.log(torch.clamp(masked_select_action_prob, min=1e-12))

                # compute entropy?
                select_entropy = - (log_select_action_prob * masked_select_action_prob).sum(1)
                chosen_log_policy_prob = log_select_action_prob.gather(1, select_unit_action)

                # Step
                state = env.step([select_action])[0]
                # if state.reward > 1:
                #     reward = np.asscalar(np.array([10]))
                # else:
                #     reward = np.asscalar(np.array([-0.2]))
                reward = np.asscalar(np.array([state.reward]))
                episode_reward += reward

                entropies.append(select_entropy)
                critic_values.append(value)
                chosen_log_policy_probs.append(chosen_log_policy_prob)
                rewards.append(reward)

                episode_done = (episode_length >= args['max_eps_length']) or state.last()
                if episode_done:
                    episode_length = 0
                    env.reset()
                    state = env.step([actions.FunctionCall(_NO_OP, [])])[0]
                    break

                # ------------------ The following is for spatial policy -------------------

                if _MOVE_SCREEN in state.observation['available_actions']:
                    screen_observation = Variable(torch.from_numpy(game_inferface.get_screen_obs(
                        timesteps=state,
                        indexes=[4, 5, 6, 7, 8, 9, 14, 15],
                    ))).cuda(gpu_id)

                    spatial_move_indicator = Variable(torch.ones(1, 1, 32, 32)).cuda(gpu_id)

                    if args['version'] == 2:
                        _, spatial_action_prob, value, _ = local_model(screen_observation, spatial_move_indicator)
                    elif args['version'] == 4:
                        _, spatial_action_prob, value, _ = local_model(screen_observation, spatial_move_indicator, 1)
                    else:
                        spatial_action_prob, value, _ = local_model(screen_observation, spatial_move_indicator)


                    spatial_action = spatial_action_prob.multinomial()
                    move_action = game_inferface.build_action(_MOVE_SCREEN, spatial_action[0].cpu())

                    log_spatial_action_prob = torch.log(torch.clamp(spatial_action_prob, min=1e-12))
                    spatial_entropy = - (log_spatial_action_prob * spatial_action_prob).sum(1)

                    chosen_log_spatial_action_prob = log_spatial_action_prob.gather(1, spatial_action)
                    chosen_log_policy_prob += chosen_log_spatial_action_prob

                    state = env.step([move_action])[0]
                    # if state.reward > 1:
                    #     reward = np.asscalar(np.array([10]))
                    # else:
                    #     reward = np.asscalar(np.array([-0.2]))
                    reward = np.asscalar(np.array([state.reward]))
                    episode_reward += reward

                    entropies.append(spatial_entropy)
                    critic_values.append(value)
                    chosen_log_policy_probs.append(chosen_log_spatial_action_prob)
                    rewards.append(reward)
                else:
                    state = env.step([actions.FunctionCall(_NO_OP, [])])[0]
                    if isinstance(state.reward, int):
                        reward = np.asscalar(np.array([state.reward]))
                    else:
                        reward = np.asscalar(state.reward)
                    # reward = np.asscalar(np.array([state.reward]))

                    rewards[-1] += reward
                    episode_reward += reward
                    with open('log.txt', 'a') as fd:
                        fd.writelines('1')

                # update statistical information
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
                ))).cuda(gpu_id)

                select_indicator = Variable(torch.zeros(1, 1, 32, 32)).cuda(gpu_id)

                if args['version'] == 2:
                    _, _, value, _ = local_model(screen_observation, select_indicator)
                elif args['version'] == 4:
                    _, _, value, _ = local_model(screen_observation, select_indicator, 0)
                else:
                    _, value, _ = local_model(screen_observation, select_indicator)

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

                policy_log_for_action = chosen_log_policy_probs[i]
                policy_loss += -(
                            policy_log_for_action * Variable(gae_ts, requires_grad=False) + 0.01 * entropies[i])


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

            if global_counter.value >= args['max_steps']:
                print("Finish worker:", worker_id)
                return