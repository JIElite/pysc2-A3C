import time

import numpy

from pysc2.agents import base_agent
from pysc2.lib import actions, features


# features
_FEATURES_SCREEN_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_FEATURES_SCREEN_SELECTTED = features.SCREEN_FEATURES.selected.index
_FEATURES_SCREEN_CREEP = features.SCREEN_FEATURES.creep.index


# player relative
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3
_PLAYER_HOSTILE = 4

# action type
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id

_NOT_QUEUED = [0]
_SELECT_ALL = [0]

i = 0


class CollectAgent(base_agent.BaseAgent):

    def step(self, obs):
        super(CollectAgent, self).step(obs)

        global i
        print(i)
        i += 1

        # if _MOVE_SCREEN in obs.observation['available_actions']:
        #     return actions.FunctionCall(_NO_OP, [])
        time.sleep(1)

        player_relative = obs.observation['screen'][_FEATURES_SCREEN_PLAYER_RELATIVE]

        neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
        player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()


        print(obs)
        return actions.FunctionCall(_NO_OP, [])


# TODO random agent

class BuiltInAgent(base_agent.BaseAgent):
    """An agent specifically for solving the CollectMineralShards map."""

    def step(self, obs):
        super(BuiltInAgent, self).step(obs)


        if _MOVE_SCREEN in obs.observation["available_actions"]:
            player_relative = obs.observation["screen"][_FEATURES_SCREEN_PLAYER_RELATIVE]
            neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
            player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
            if not neutral_y.any() or not player_y.any():
                return actions.FunctionCall(_NO_OP, [])
            player = [int(player_x.mean()), int(player_y.mean())]
            closest, min_dist = None, None
            for p in zip(neutral_x, neutral_y):
                dist = numpy.linalg.norm(numpy.array(player) - numpy.array(p))
                if not min_dist or dist < min_dist:
                    closest, min_dist = p, dist
            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, closest])
        else:
            return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])