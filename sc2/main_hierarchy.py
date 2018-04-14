"""
This main is used to whole task for starcraft minigame
There are three sub tasks:
1. collect mineral shards
2. defeat building
3. train marine
"""


import sys
import os
os.environ['OMP_NUM_THREADS']='1'

from absl import app, flags
from pysc2 import maps
from pysc2.env import sc2_env
import torch
import torch.multiprocessing as mp

from model import FullyConv
from optimizer import SharedAdam
from monitor import evaluator
from train_hierarchical import train_master

FLAGS = flags.FLAGS
# Game related settings
flags.DEFINE_string("map", "CollectMineralShardsFiveExtended", "Name of a map to use.")
flags.DEFINE_integer("screen_resolution", 32, "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 32, "Resolution for minimap feature layers.")
flags.DEFINE_bool("visualize", False, "Whether to render with pygame.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_enum("agent_race", None, sc2_env.races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", None, sc2_env.races.keys(), "Bot's race.")
flags.DEFINE_integer('worker_steps', 10500000, "steps run for each worker")
flags.DEFINE_integer('max_eps_length', 5000, "max length run for each episode")

# Learning related settings
flags.DEFINE_float("learning_rate", 5e-4, "Learning rate for training.")
flags.DEFINE_float("gamma", 0.99, "Discount rate for future rewards.")
flags.DEFINE_integer("num_of_workers", 8, "How many instances to run in parallel.")
flags.DEFINE_integer("n_steps", 8,  "How many steps do we compute the Return (TD)")
flags.DEFINE_integer("seed", 5, "torch random seed")
flags.DEFINE_float("tau", 1.0, "tau for GAE")
flags.DEFINE_integer("gpu", 0, "gpu device")


FLAGS(sys.argv)


def main(argv):
    # global settings
    mp.set_start_method('spawn')
    maps.get(FLAGS.map)
    global_counter = mp.Value('i', 0)
    summary_queue = mp.Queue()

    # share model
    master_model = FullyConv(screen_channels=8, screen_resolution=(FLAGS.screen_resolution, FLAGS.screen_resolution)).cuda(FLAGS.gpu)
    master_model.share_memory()
    sub_model = FullyConv(screen_channels=8, screen_resolution=(FLAGS.screen_resolution, FLAGS.screen_resolution)).cuda(FLAGS.gpu)
    sub_model.load_state_dict(torch.load('./models/task1_extended_15264524/model_best'))
    sub_model.share_memory()

    optimizer = SharedAdam(master_model.parameters(), lr=FLAGS.learning_rate)
    optimizer.share_memory()

    shared_models = [master_model, sub_model]
    worker_list = []
    evaluate_worker = mp.Process(target=evaluator, args=(summary_queue, master_model, optimizer,
                                                         global_counter,
                                                         FLAGS.num_of_workers * FLAGS.worker_steps,
                                                         'mask'))
    evaluate_worker.start()
    worker_list.append(evaluate_worker)

    for worker_id in range(FLAGS.num_of_workers):
        worker = mp.Process(target=train_master, args=(worker_id, FLAGS.flag_values_dict(), shared_models, optimizer, global_counter, summary_queue))
        worker.start()
        worker_list.append(worker)

    for worker in worker_list:
        worker.join()


if __name__ == '__main__':
    app.run(main)