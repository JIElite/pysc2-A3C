import time
import numpy as np
import torch
from torch.autograd import Variable

from pysc2.lib import actions
from pysc2.lib import features

from envs import GameInterfaceHandler
from envs import create_pysc2_env
from model2 import FullyConvSelectPolicy
from optimizer import ensure_shared_grad
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


def train_selection_policy(worker_id, args, shared_model, optimizer, global_counter, summary_queue):
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
        shared_selection_model = shared_model[0]
        local_collect_model = shared_model[1]
        local_destroy_model = shared_model[2]
        local_selection_model = FullyConvSelectPolicy(screen_channels=8, screen_resolution=[args['screen_resolution']]*2).cuda(args['gpu'])

        env.reset()
        state = env.step([actions.FunctionCall(_NO_OP, [])])[0]
        episode_done = False
        episode_length = 0
        episode_reward = 0

        while True:
            # Sync the parameters with shared model
            local_selection_model.load_state_dict(shared_selection_model.state_dict())

            # Reset n-step experience buffer
            # entropies = []
            critic_values = []
            spatial_policy_log_probs = []
            rewards = []


            # step forward n steps
            for step in range(args['n_steps']):
                screen_observation = Variable(torch.from_numpy(game_inferface.get_screen_obs(
                                                                    timesteps=state,
                                                                    indexes=[4, 5, 6, 7, 8, 9, 14, 15],
                                                                ))).cuda()

                select_unit_action_prob, value, selected_task = local_selection_model(screen_observation)

                # select unit
                selection_mask = torch.from_numpy((state.observation['screen'][_SCREEN_PLAYER_RELATIVE] == 1).astype('float32'))
                selection_mask = Variable(selection_mask.view(1, -1), requires_grad=False).cuda()
                masked_select_unit_action_prob = select_unit_action_prob * selection_mask

                if float(masked_select_unit_action_prob.sum().cpu().data.numpy()) < 1e-12:
                    # print('normalize')
                    masked_select_unit_action_prob += 1.0 * selection_mask
                    masked_select_unit_action_prob /= masked_select_unit_action_prob.sum()
                    # print('mask matrix:', selection_mask)
                else:
                    masked_select_unit_action_prob = masked_select_unit_action_prob / masked_select_unit_action_prob.sum()

                # print('sum of masked action prob:', masked_select_unit_action_prob.sum())
                # print('-------------------------------------------------------------------------------')
                try:
                    select_action = masked_select_unit_action_prob.multinomial()
                except:
                    print("Error detect!")
                    print(masked_select_unit_action_prob)

                # print(select_action)

                # select task type
                task = selected_task.multinomial()

                log_select_spatial_action_prob = torch.log(torch.clamp(masked_select_unit_action_prob, min=1e-12))
                log_select_task_prob = torch.log(torch.clamp(selected_task, min=1e-12))

                # TODO verify derivative of this entropy
                # select_entropy = - (log_select_spatial_action_prob * masked_select_unit_action_prob).sum(1)
                # task_entropy = - (log_select_task_prob * selected_task).sum()
                # entropy = select_entropy + task_entropy


                chosen_unit_selection_log_action_prob = log_select_spatial_action_prob.gather(1, select_action)
                chosen_task_selection_action_log_prob = log_select_task_prob[task]
                chosen_action_log_prob = chosen_unit_selection_log_action_prob + chosen_task_selection_action_log_prob

                # record n-step experience
                # entropies.append(entropy)
                spatial_policy_log_probs.append(chosen_action_log_prob)
                critic_values.append(value)

                # Step
                action = game_inferface.build_action(_SELECT_POINT, select_action[0].cpu())
                state = env.step([action])[0]

                if state.reward > 1:
                    reward = np.asscalar(np.array([10]))
                else:
                    reward = np.asscalar(np.array([-0.2]))
                rewards.append(reward)
                episode_reward += reward

                episode_done = (episode_length >= args['max_eps_length']) or state.last()
                if episode_done:
                    episode_length = 0
                    env.reset()
                    state = env.step([actions.FunctionCall(_NO_OP, [])])[0]
                    break


                task = int(task.cpu().data.numpy())
                if task == 0 and _MOVE_SCREEN in state.observation['available_actions']:
                    # collection mineral shards
                    # TODO volatile
                    screen_observation = Variable(torch.from_numpy(game_inferface.get_screen_obs(
                        timesteps=state,
                        indexes=[4, 5, 6, 7, 8, 9, 14, 15],
                    )), volatile=True).cuda()

                    spatial_action_prob, value, _ = local_collect_model(screen_observation)
                    spatial_action = spatial_action_prob.multinomial()

                    action = game_inferface.build_action(_MOVE_SCREEN, spatial_action[0].cpu())
                    state = env.step([action])[0]
                    # if state.reward > 1:
                    #     reward = np.asscalar(np.array([10]))
                    # else:
                    #     reward = np.asscalar(np.array([-0.2]))

                    # TODO learning
                    # log_prob = torch.log(torch.clamp(spatial_action_prob, min=1e-12))
                    # chosen_log_prob = log_prob.gather(1, spatial_action)
                    # spatial_policy_log_probs.append(chosen_log_prob)
                    # critic_values.append(value)
                    # rewards.append(reward)

                elif task == 1 and _RIGHT_CLICK in state.observation['available_actions']:
                    # destroy enemy's buildings
                    # TODO volatile
                    screen_observation = Variable(torch.from_numpy(game_inferface.get_screen_obs(
                        timesteps=state,
                        indexes=[4, 5, 6, 7, 8, 9, 14, 15],
                    )), volatile=True).cuda()

                    destroy_action_prob, value, _ = local_destroy_model(screen_observation)
                    destroy_position = destroy_action_prob.multinomial()

                    action = game_inferface.build_action(_RIGHT_CLICK, destroy_position[0].cpu())
                    state = env.step([action])[0]
                    # if state.reward > 1:
                    #     reward = np.asscalar(np.array([10]))
                    # else:
                    #     reward = np.asscalar(np.array([-0.2]))

                    # TODO learning
                    # log_prob = torch.log(torch.clamp(destroy_action_prob, min=1e-12))
                    # chosen_log_prob = log_prob.gather(1, destroy_position)
                    # spatial_policy_log_probs.append(chosen_log_prob)
                    # critic_values.append(value)
                    # rewards.append(reward)

                else:
                    action = actions.FunctionCall(_NO_OP, [])
                    state = env.step([action])[0]

                    # TODO learning
                    # if state.reward > 1:
                    #     reward = np.asscalar(np.array([10]))
                    # else:
                    #     reward = np.asscalar(np.array([-0.2]))
                    # rewards[-1] += reward

                # TODO unlearning
                if state.reward > 1:
                    reward += np.asscalar(np.array([10]))
                else:
                    reward += np.asscalar(np.array([-0.2]))

                rewards[-1] += reward
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
                _, value, _ = local_selection_model(screen_observation)
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
                # policy_loss += -(spatial_policy_log_probs[i] * Variable(gae_ts, requires_grad=False) + 0.05 *entropies[i])
                policy_loss += -(spatial_policy_log_probs[i] * Variable(gae_ts, requires_grad=False))

            optimizer.zero_grad()
            total_loss = policy_loss + 0.5 * value_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm(local_selection_model.parameters(), 40)
            ensure_shared_grad(local_selection_model, shared_selection_model)
            optimizer.step()

            if episode_done:
                summary_queue.put((global_counter.value, episode_reward))
                episode_reward = 0

            if global_counter.value >= args['max_steps']:
                print("Finish worker:", worker_id)
                return