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
from model2 import FullyConvSelectPolicy
from optimizer import SharedAdam
from monitor import evaluator
from train_hierarchy_collect_and_destroy import train_selection_policy

FLAGS = flags.FLAGS
# Game related settings
flags.DEFINE_string("map", "CollectAndDestroy", "Name of a map to use.")
flags.DEFINE_integer("screen_resolution", 32, "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 32, "Resolution for minimap feature layers.")
flags.DEFINE_bool("visualize", False, "Whether to render with pygame.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_enum("agent_race", None, sc2_env.races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", None, sc2_env.races.keys(), "Bot's race.")
flags.DEFINE_integer('max_steps', 10500000, "steps run for each worker")
flags.DEFINE_integer('max_eps_length', 5000, "max length run for each episode")

# Learning related settings
flags.DEFINE_float("learning_rate", 5e-4, "Learning rate for training.")
flags.DEFINE_float("gamma", 0.99, "Discount rate for future rewards.")
flags.DEFINE_integer("num_of_workers", 8, "How many instances to run in parallel.")
flags.DEFINE_integer("n_steps", 8,  "How many steps do we compute the Return (TD)")
flags.DEFINE_integer("seed", 5, "torch random seed")
flags.DEFINE_float("tau", 1.0, "tau for GAE")
flags.DEFINE_integer("gpu", 0, "gpu device")
flags.DEFINE_string("postfix", "", "postfix of training data")

FLAGS(sys.argv)

torch.cuda.set_device(FLAGS.gpu)
print("CUDA device:", torch.cuda.current_device())


def main(argv):
    # global settings
    mp.set_start_method('spawn')
    maps.get(FLAGS.map)
    global_counter = mp.Value('i', 0)
    summary_queue = mp.Queue()

    # shared model
    selection_model = FullyConvSelectPolicy(screen_channels=8, screen_resolution=(FLAGS.screen_resolution, FLAGS.screen_resolution)).cuda()
    selection_model.share_memory()

    collect_model = FullyConv(screen_channels=8, screen_resolution=(FLAGS.screen_resolution, FLAGS.screen_resolution)).cuda()
    collect_model.load_state_dict(torch.load('./models/task1_300s_original_16347668/model_best'))
    collect_model.share_memory()

    destroy_model = FullyConv(screen_channels=8, screen_resolution=(FLAGS.screen_resolution, FLAGS.screen_resolution)).cuda()
    destroy_model.load_state_dict(torch.load('./destroy_buildings/model_latest_defeat2buildings_no_op2_2'))
    destroy_model.share_memory()

    optimizer = SharedAdam(selection_model.parameters(), lr=FLAGS.learning_rate)
    optimizer.share_memory()

    shared_models = [selection_model, collect_model, destroy_model]
    worker_list = []
    evaluate_worker = mp.Process(target=evaluator, args=(summary_queue, selection_model, optimizer,
                                                         global_counter,
                                                         FLAGS.max_steps,
                                                         FLAGS.postfix))
    evaluate_worker.start()
    worker_list.append(evaluate_worker)
    for worker_id in range(FLAGS.num_of_workers):
        worker = mp.Process(target=train_selection_policy, args=(worker_id, FLAGS.flag_values_dict(), shared_models, optimizer, global_counter, summary_queue))
        worker.start()
        worker_list.append(worker)

    for worker in worker_list:
        worker.join()


if __name__ == '__main__':
    app.run(main)