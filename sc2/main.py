import sys
import os
#os.environ['OMP_NUM_THREADS']='1'

from absl import app, flags
from pysc2 import maps
from pysc2.env import sc2_env
import torch
import torch.multiprocessing as mp

from model import FullyConv, FullyConvExtended, FullyConvBN
from optimizer import SharedAdam
from worker import worker_fn
from monitor import evaluator


FLAGS = flags.FLAGS
# Game related settings
flags.DEFINE_string("map", "CollectMineralShardsSingleExtended", "Name of a map to use.")
flags.DEFINE_integer("screen_resolution", 32, "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 32, "Resolution for minimap feature layers.")
flags.DEFINE_bool("visualize", False, "Whether to render with pygame.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_enum("agent_race", None, sc2_env.races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", None, sc2_env.races.keys(), "Bot's race.")
flags.DEFINE_integer('max_steps', 10010000, "steps run for each worker")
flags.DEFINE_integer('max_eps_length', 5000, "max length run for each episode")

# Learning related settings
flags.DEFINE_float("learning_rate", 5e-4, "Learning rate for training.")
flags.DEFINE_float("gamma", 0.99, "Discount rate for future rewards.")
flags.DEFINE_integer("num_of_workers", 8, "How many instances to run in parallel.")
flags.DEFINE_integer("n_steps", 8,  "How many steps do we compute the Return (TD)")
flags.DEFINE_integer("seed", 5, "torch random seed")
flags.DEFINE_float("tau", 1.0, "tau for GAE")
flags.DEFINE_boolean("extend_model", False, "extend conv3 or not")
flags.DEFINE_integer("gpu", 0, "gpu device")
flags.DEFINE_string("postfix", "", "postfix of training data")
flags.DEFINE_integer("insert_no_op_steps", 0, "num of steps to insert NO_OP between each agent steps")
flags.DEFINE_boolean('use_bn', False, 'use batch norm or not')
FLAGS(sys.argv)

torch.cuda.set_device(FLAGS.gpu)
print("CUDA device:", torch.cuda.current_device())


def main(argv):
    # global settings
    mp.set_start_method('spawn')
    maps.mini_games.mini_games.append("CollectMineralShardsSingleExtended")
    maps.get(FLAGS.map)
    global_counter = mp.Value('i', 0)
    summary_queue = mp.Queue()

    # share model
    if FLAGS.use_bn:
        if FLAGS.extend_model:
            model = None
        else:
            model = FullyConvBN
    else:
        if FLAGS.extend_model:
            model = FullyConvExtended
        else:
            model = FullyConv
    print(model)

    shared_model = model(screen_channels=8, screen_resolution=(FLAGS.screen_resolution, FLAGS.screen_resolution)).cuda()
    shared_model.train()

    shared_model.share_memory()
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
        worker = mp.Process(target=worker_fn, args=(worker_id, FLAGS.flag_values_dict(), shared_model, optimizer, global_counter, summary_queue))
        worker.start()
        worker_list.append(worker)

    for worker in worker_list:
        worker.join()


if __name__ == '__main__':
    app.run(main)
