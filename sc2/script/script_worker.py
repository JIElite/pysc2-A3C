import time

import numpy as np
import torch
from torch.autograd import Variable

from pysc2.lib import actions
from pysc2.lib import features

from envs import GameInterfaceHandler
from envs import create_pysc2_env


# action id
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
select_army = [actions.FunctionCall(_SELECT_ARMY, [[0]])]

# action parameters
_NOT_QUEUED = []

# Feature id
_SCREEN_SELECTED = features.SCREEN_FEATURES.selected.index
_SCREEN_PLAYER_ID = features.SCREEN_FEATURES.player_id.index
_SCREEN_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index


def worker_fn(worker_id, agent, args, global_counter, summary_queue):
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
    with env:
        env.reset()
        state = env.step(actions=select_army)[0]
        episode_done = False
        episode_length = 0
        episode_reward = 0

        while True:
            # step forward n steps
            for step in range(args['n_steps']):
                action = agent.step(state)

                state = env.step([action])[0]
                reward = state.reward
                episode_reward += reward

                episode_length += 1
                global_counter.value += 1

                episode_done = (episode_length >= args['max_eps_length']) or state.last()
                if episode_done:
                    episode_length = 0
                    env.reset()
                    state = env.step(actions=select_army)[0]
                    break


            if episode_done:
                summary_queue.put((global_counter.value, episode_reward))
                episode_reward = 0

