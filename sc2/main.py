import sys
import os
os.environ['OMP_NUM_THREADS']='1'

from absl import app, flags
from pysc2 import maps
from pysc2.env import sc2_env
import torch.multiprocessing as mp

from model import FullyConv
from optimizer import SharedAdam
from worker import worker_fn
from monitor import evaluator, process_statistics


FLAGS = flags.FLAGS
# Game related settings
flags.DEFINE_string("map", "DefeatBuilding", "Name of a map to use.")
flags.DEFINE_integer("screen_resolution", 32, "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 32, "Resolution for minimap feature layers.")
flags.DEFINE_bool("visualize", False, "Whether to render with pygame.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_enum("agent_race", None, sc2_env.races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", None, sc2_env.races.keys(), "Bot's race.")
flags.DEFINE_integer('total_steps', 10500000, "steps run for each worker")
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
    global_eps = mp.Value('i', 0)
    global_avg_perf = mp.Value('d', 0.0)
    global_recently_best_avg = mp.Value('d', 0.0)

    summary_queue = mp.Queue()

    # share model
    shared_model = FullyConv(screen_channels=8, screen_resolution=(FLAGS.screen_resolution, FLAGS.screen_resolution)).cuda()
    shared_model.share_memory()
    optimizer = SharedAdam(shared_model.parameters(), lr=FLAGS.learning_rate)
    optimizer.share_memory()

    worker_list = []

    # Evaluation-related worker
    statistics_worker = mp.Process(target=process_statistics, args=(
        summary_queue, global_eps, global_avg_perf, global_recently_best_avg
    ))
    statistics_worker.start()
    worker_list.append(statistics_worker)

    evaluate_worker = mp.Process(target=evaluator, args=(
        shared_model, global_counter, global_eps, global_avg_perf, global_recently_best_avg
    ))
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
