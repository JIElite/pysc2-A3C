import sys
import os
os.environ['OMP_NUM_THREADS']='1'

from absl import app, flags
from pysc2 import maps
from pysc2.env import sc2_env
import torch.multiprocessing as mp

from script.collect import BuiltInAgent
from script.script_worker import worker_fn
from script.monitor import evaluator


FLAGS = flags.FLAGS
# Game related settings
flags.DEFINE_string("map", "CollectMineralShardsSingle", "Name of a map to use.")
flags.DEFINE_integer("screen_resolution", 64, "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64, "Resolution for minimap feature layers.")
flags.DEFINE_bool("visualize", False, "Whether to render with pygame.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_enum("agent_race", None, sc2_env.races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", None, sc2_env.races.keys(), "Bot's race.")
flags.DEFINE_integer('total_steps', 100000000, "steps run for each worker")
flags.DEFINE_integer('max_eps_length', 3000, "max length run for each episode")

# Learning related settings
flags.DEFINE_float("learning_rate", 5e-4, "Learning rate for training.")
flags.DEFINE_float("gamma", 0.99, "Discount rate for future rewards.")
flags.DEFINE_integer("num_of_workers", 4, "How many instances to run in parallel.")
flags.DEFINE_integer("n_steps", 8,  "How many steps do we compute the Return (TD)")
flags.DEFINE_integer("seed", 5, "torch random seed")
flags.DEFINE_float("tau", 1.0, "tau for GAE")

FLAGS(sys.argv)


def main(argv):
    # global settings
    maps.get(FLAGS.map)
    global_counter = mp.Value('i', 0)
    summary_queue = mp.Queue()
    script_collect_agent = BuiltInAgent()

    worker_list = []
    evaluate_worker = mp.Process(target=evaluator, args=(summary_queue,))
    evaluate_worker.daemon = True
    evaluate_worker.start()
    worker_list.append(evaluate_worker)

    for i in range(FLAGS.num_of_workers):
        worker = mp.Process(target=worker_fn, args=(i, script_collect_agent, FLAGS.flag_values_dict(), global_counter, summary_queue))
        worker.daemon = True
        worker.start()
        worker_list.append(worker)

    for worker in worker_list:
        worker.join()


if __name__ == '__main__':
    app.run(main)