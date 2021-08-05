from itertools import count
import sys
import time

from absl import app, flags
from pysc2.env import sc2_env
from pysc2.lib import actions

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.distributions import Categorical

from envs import create_pysc2_env, GameInterfaceHandler
from model import FullyConv


FLAGS = flags.FLAGS
# Game related settings
flags.DEFINE_string(
    "map", "CollectMineralShardsSingleExtended", "Name of a map to use."
)
flags.DEFINE_integer("screen_resolution", 32, "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 32, "Resolution for minimap feature layers.")
flags.DEFINE_bool("visualize", False, "Whether to render with pygame.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_enum("agent_race", None, sc2_env.races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", None, sc2_env.races.keys(), "Bot's race.")
flags.DEFINE_integer("total_steps", 10500000, "steps run for each worker")
flags.DEFINE_integer("max_eps_length", 5000, "max length run for each episode")

# Agent related settings
flags.DEFINE_integer("seed", 5, "torch random seed")
flags.DEFINE_integer(
    "num_episodes", 100, "# of episode for agent to play with environment"
)
flags.DEFINE_bool("use_gpu", False, "Use gpu or not")
flags.DEFINE_integer("gpu", 0, "gpu device")
flags.DEFINE_string("pretrained_model", "", "path of pretrained model")
flags.DEFINE_float("step_delay", 0.1, "delay for each agent step")
flags.DEFINE_integer(
    "insert_no_op_steps", 0, "num of steps to insert NO_OP between each agent steps"
)
flags.DEFINE_bool("display_policy", False, "display policy heatmap or not")
FLAGS(sys.argv)


## XXX What is this?
torch.set_printoptions(5000)

device = torch.device(
    f"cuda:{FLAGS.gpu}" if torch.cuda.is_available() and FLAGS.use_gpu else "cpu"
)
print("Infernce device:", device)

# action type id
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_RIGHT_CLICK = actions.FUNCTIONS.Smart_screen.id


def main(argv):
    torch.manual_seed(FLAGS.seed)

    # build environment
    env_args = {
        "map_name": FLAGS.map,
        "agent_race": FLAGS.agent_race,
        "bot_race": FLAGS.bot_race,
        "step_mul": FLAGS.step_mul,
        "screen_size_px": [FLAGS.screen_resolution] * 2,
        "minimap_size_px": [FLAGS.minimap_resolution] * 2,
        "visualize": FLAGS.visualize,
    }
    env = create_pysc2_env(env_args)
    game_inferface = GameInterfaceHandler(
        screen_resolution=FLAGS.screen_resolution,
        minimap_resolution=FLAGS.minimap_resolution,
    )

    # Set up action type
    action_id = _NO_OP
    if "CollectMineralShardsSingle" in FLAGS.map:
        action_id = _MOVE_SCREEN
    elif "Defeat" in FLAGS.map:
        action_id = _RIGHT_CLICK

    # TODO can we load full model? (includes model architecture)
    # TODO device
    # agent's model
    model = FullyConv(
        screen_channels=8,
        screen_resolution=(FLAGS.screen_resolution, FLAGS.screen_resolution),
    )
    model = model.to(device)
    model.load_state_dict(torch.load(FLAGS.pretrained_model, map_location=device))

    # play with environment
    total_eps_return = 0
    with env:
        for i_episode in range(FLAGS.num_episodes):
            env.reset()
            state = env.step([actions.FunctionCall(_NO_OP, [])])[0]
            episodic_reward = 0

            for step in count():
                # Extract features
                screen_observation = torch.from_numpy(
                    game_inferface.get_screen_obs(
                        timesteps=state, indexes=[4, 5, 6, 7, 8, 9, 14, 15]
                    )
                )
                screen_observation = screen_observation.to(device)

                # Run Model Inference
                with torch.no_grad():
                    log_spatial_action_prob, _, spatial_vis = model(screen_observation)

                # Sample action from policy
                action_sampler = Categorical(torch.exp(log_spatial_action_prob))
                spatial_action = action_sampler.sample().to("cpu")[0]

                # Take action
                action = game_inferface.build_action(action_id, spatial_action)
                state = env.step([action])[0]
                reward = np.asscalar(state.reward)
                episodic_reward += reward
                time.sleep(FLAGS.step_delay)

                if FLAGS.display_policy and step % 5 == 0:
                    np_spatial = spatial_vis.data.cpu().numpy()
                    np_spatial = np.clip(np_spatial, 0.03126, 1)
                    sns.heatmap(np_spatial, linewidths=0.5, cmap="YlGnBu")
                    plt.show()
                    print(np_spatial)
                    input()

                episode_done = (step >= FLAGS.max_eps_length) or state.last()
                for no_op_step in range(FLAGS.insert_no_op_steps):
                    action = actions.FunctionCall(_NO_OP, [])
                    state = env.step([action])[0]
                    reward += state.reward.astype(np.float32)
                    episodic_reward += reward
                    time.sleep(FLAGS.step_delay)

                if episode_done:
                    print("Episodic reward:", episodic_reward)
                    total_eps_return += episodic_reward
                    break

    print("mean eps return:", total_eps_return / FLAGS.num_episodes)


if __name__ == "__main__":
    app.run(main)
