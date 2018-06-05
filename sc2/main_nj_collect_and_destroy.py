"""
This main is used to whole task for starcraft minigame
There are three sub tasks:
1. collect mineral shards
2. defeat building
3. train marine
"""


import sys
import os
import time
os.environ['OMP_NUM_THREADS']='1'

from absl import app, flags
from pysc2 import maps
from pysc2.env import sc2_env
import torch
import torch.multiprocessing as mp

from model import FullyConv
from model2 import CollectAndDestroyGraftingNet, CollectAndDestroyGraftingDropoutNet, CollectAndDestroyBaseline
from optimizer import SharedAdam
from monitor import evaluator
from train_nj_collect_and_destroy import train_policy

FLAGS = flags.FLAGS
# Game related settings
flags.DEFINE_string("map", "CollectAndDestroyAirSCV", "Name of a map to use.")
flags.DEFINE_integer("screen_resolution", 48, "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 48, "Resolution for minimap feature layers.")
flags.DEFINE_bool("visualize", False, "Whether to render with pygame.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_enum("agent_race", None, sc2_env.races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", None, sc2_env.races.keys(), "Bot's race.")
flags.DEFINE_integer('max_steps', 10500000, "steps run for each worker")
flags.DEFINE_integer('max_eps_length', 5000, "max length run for each episode")

# Learning related settings
flags.DEFINE_float("learning_rate", 5e-5, "Learning rate for training.")
flags.DEFINE_float("gamma", 0.99, "Discount rate for future rewards.")
flags.DEFINE_integer("num_of_workers", 8, "How many instances to run in parallel.")
flags.DEFINE_integer("n_steps", 8,  "How many steps do we compute the Return (TD)")
flags.DEFINE_integer("seed", 5, "torch random seed")
flags.DEFINE_float("tau", 1.0, "tau for GAE")
flags.DEFINE_integer("gpu", 0, "gpu device")
flags.DEFINE_integer("version", 0, "version of network")
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
    if FLAGS.version == 0:
        model = CollectAndDestroyGraftingNet
    elif FLAGS.version == 1:
        model = CollectAndDestroyGraftingDropoutNet
    elif FLAGS.version == 2:
        model = CollectAndDestroyBaseline
    print("model type:", model)

    shared_model = model(screen_channels=8, screen_resolution=(FLAGS.screen_resolution, FLAGS.screen_resolution)).cuda()
    shared_model.share_memory()
    shared_model.train()

    if FLAGS.version != 2:
        collect_model = FullyConv(screen_channels=8, screen_resolution=(FLAGS.screen_resolution, FLAGS.screen_resolution)).cuda()
        collect_model.load_state_dict(torch.load('./collect_task_scv/model_latest_collect_scv_res48_no_op6'))
        #
        destroy_model = FullyConv(screen_channels=8, screen_resolution=(FLAGS.screen_resolution, FLAGS.screen_resolution)).cuda()
        destroy_model.load_state_dict(torch.load('./defeat_buildings_banshee/model_latest_defeat4buildings_res48'))

        shared_model.conv_collect.load_state_dict(collect_model.conv1.state_dict())
        shared_model.conv_destroy.load_state_dict(destroy_model.conv1.state_dict())
        shared_model.collect_policy.load_state_dict(collect_model.spatial_policy.state_dict())
        shared_model.destroy_policy.load_state_dict(destroy_model.spatial_policy.state_dict())

    optimizer = SharedAdam(shared_model.parameters(), lr=FLAGS.learning_rate)
    optimizer.share_memory()

    worker_list = []
    evaluate_worker = mp.Process(target=evaluator, args=(summary_queue, shared_model, optimizer,
                                                         global_counter,
                                                         FLAGS.max_steps,
                                                         FLAGS.postfix))
    evaluate_worker.start()
    worker_list.append(evaluate_worker)
    for worker_id in range(FLAGS.num_of_workers):
        worker = mp.Process(target=train_policy, args=(worker_id, FLAGS.flag_values_dict(), shared_model, optimizer, global_counter, summary_queue))
        worker.start()
        worker_list.append(worker)
        time.sleep(0.5)

    for worker in worker_list:
        worker.join()


if __name__ == '__main__':
    app.run(main)