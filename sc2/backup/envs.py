import numpy as np

from pysc2.env import sc2_env
from pysc2.env import available_actions_printer
from pysc2.lib import features
from pysc2.lib import actions


TOTAL_SCREEN_LAYERS = 17


def create_pysc2_env(env_args):
    env = sc2_env.SC2Env(**env_args)
    env = available_actions_printer.AvailableActionsPrinter(env)
    return env


class GameInterfaceHandler(object):

    def __init__(self, screen_resolution, minimap_resolution):
        self.dtype = np.float32
        self.screen_player_id = features.SCREEN_FEATURES.player_id.index
        self.screen_unit_type = features.SCREEN_FEATURES.unit_type.index
        self.screen_resolution = screen_resolution
        self.minimap_resolution = minimap_resolution

    def get_screen_obs(self, timesteps, indexes=None):
        '''
        You can filter the observation(timesteps.observation) using indexes

        :param timesteps: env timesteps
        :param indexes: mask for timesteps which is used to select feature layers to use
        :return: preprocessed_screen_observation: np.array (layer, screen_resolution, screen_resolution)
        '''
        layers = []
        screen_obs = np.array(timesteps.observation['screen'], dtype=np.float32)

        if indexes == None:
            indexes = [i for i in range(TOTAL_SCREEN_LAYERS)]

        for i in indexes:
            if i == self.screen_player_id or i == self.screen_unit_type:
                layers.append(screen_obs[i] / features.SCREEN_FEATURES[i].scale)
            elif features.SCREEN_FEATURES[i].type == features.FeatureType.SCALAR:
                layers.append(screen_obs[i] / features.SCREEN_FEATURES[i].scale)
            else:
                layers.append(screen_obs[i])

        preprocessed_screen_obs = np.stack(layers, axis=0)
        return np.expand_dims(preprocessed_screen_obs, axis=0)

    def build_action(self, action_id, spatial_action):
        target = spatial_action.data.numpy()

        target_point = [
            int(target % self.screen_resolution),
            int(target // self.screen_resolution)
        ]

        act_args = []
        for arg in actions.FUNCTIONS[action_id].args:
            if arg.name in ('screen', 'minimap', 'screen2'):
                act_args.append(target_point)
            else:
                act_args.append([0])

        return actions.FunctionCall(action_id, act_args)