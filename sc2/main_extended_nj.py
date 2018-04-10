import sys
import os
os.environ['OMP_NUM_THREADS']='1'

from absl import app, flags
from pysc2 import maps
from pysc2.env import sc2_env
import torch
import torch.multiprocessing as mp

from model import FullyConv, FullyConvExtended
from optimizer import SharedAdam
from train_neural_junction import train_extended_nj
from monitor import evaluator
from utils import freeze_layers


FLAGS = flags.FLAGS
# Game related settings
flags.DEFINE_string("map", "CollectMineralShardsSingleExtended", "Name of a map to use.")
flags.DEFINE_integer("screen_resolution", 32, "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 32, "Resolution for minimap feature layers.")
flags.DEFINE_bool("visualize", False, "Whether to render with pygame.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_enum("agent_race", None, sc2_env.races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", None, sc2_env.races.keys(), "Bot's race.")
flags.DEFINE_integer('worker_steps', 12600000, "steps run for each worker")
flags.DEFINE_integer('max_eps_length', 5000, "max length run for each episode")

# Learning related settings
flags.DEFINE_float("learning_rate", 5e-4, "Learning rate for training.")
flags.DEFINE_float("gamma", 0.99, "Discount rate for future rewards.")
flags.DEFINE_integer("num_of_workers", 8, "How many instances to run in parallel.")
flags.DEFINE_integer("n_steps", 8,  "How many steps do we compute the Return (TD)")
flags.DEFINE_integer("seed", 5, "torch random seed")
flags.DEFINE_float("tau", 1.0, "tau for GAE")

FLAGS(sys.argv)


def main(argv):
    # global settings
    mp.set_start_method('spawn')
    maps.get(FLAGS.map)
    global_counter = mp.Value('i', 0)
    summary_queue = mp.Queue()

    # share model
    shared_model = FullyConvExtended(screen_channels=8, screen_resolution=(FLAGS.screen_resolution, FLAGS.screen_resolution)).cuda()
    pretrained_model = FullyConv(screen_channels=8, screen_resolution=(FLAGS.screen_resolution, FLAGS.screen_resolution)).cuda()
    pretrained_model.load_state_dict(torch.load('./models/task1_extended_10077037/model_best'))

    shared_model.conv1.load_state_dict(pretrained_model.conv1.state_dict())
    shared_model.spatial_policy.load_state_dict(pretrained_model.spatial_policy.state_dict())
    shared_model.non_spatial_branch.load_state_dict(pretrained_model.non_spatial_branch.state_dict())
    shared_model.value.load_state_dict(pretrained_model.value.state_dict())

    freeze_layers(shared_model.conv1)
    freeze_layers(shared_model.spatial_policy)
    freeze_layers(shared_model.non_spatial_branch)
    freeze_layers(shared_model.value)


    shared_model.share_memory()
    parameters = list(shared_model.conv2.parameters()) + list(shared_model.conv3.parameters())
    optimizer = SharedAdam(parameters, lr=FLAGS.learning_rate)
    optimizer.share_memory()

    worker_list = []
    evaluate_worker = mp.Process(target=evaluator, args=(summary_queue, shared_model, optimizer,
                                                         global_counter, FLAGS.num_of_workers*FLAGS.worker_steps))
    evaluate_worker.start()
    worker_list.append(evaluate_worker)

    for worker_id in range(FLAGS.num_of_workers):
        worker = mp.Process(target=train_extended_nj, args=(worker_id, FLAGS.flag_values_dict(), shared_model, optimizer, global_counter, summary_queue))
        worker.start()
        worker_list.append(worker)

    for worker in worker_list:
        worker.join()


if __name__ == '__main__':
    app.run(main)
