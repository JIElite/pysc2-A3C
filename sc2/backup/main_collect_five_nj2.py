import sys
import os
os.environ['OMP_NUM_THREADS']='1'

from absl import app, flags
from pysc2 import maps
from pysc2.env import sc2_env
import torch
import torch.multiprocessing as mp

from model import (
    FullyConv,
)
from model2 import MultiFeaturesGrafting
from optimizer import SharedAdam
from monitor import evaluator
from train_collect_five_nj2 import train_conjunction_with_action_features
from utils import freeze_layers


FLAGS = flags.FLAGS
# Game related settings
flags.DEFINE_string("map", "CollectMineralShardsFiveExtended", "Name of a map to use.")
flags.DEFINE_integer("screen_resolution", 32, "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 32, "Resolution for minimap feature layers.")
flags.DEFINE_bool("visualize", False, "Whether to render with pygame.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_enum("agent_race", None, sc2_env.races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", None, sc2_env.races.keys(), "Bot's race.")
flags.DEFINE_integer('max_steps', 10005000, "steps run for each worker")
flags.DEFINE_integer('max_eps_length', 5000, "max length run for each episode")

# Learning related settings
flags.DEFINE_float("learning_rate", 5e-4, "Learning rate for training.")
flags.DEFINE_float("gamma", 0.99, "Discount rate for future rewards.")
flags.DEFINE_integer("num_of_workers", 8, "How many instances to run in parallel.")
flags.DEFINE_integer("n_steps", 8,  "How many steps do we compute the Return (TD)")
flags.DEFINE_integer("seed", 5, "torch random seed")
flags.DEFINE_float("tau", 1.0, "tau for GAE")
flags.DEFINE_boolean("multiple_gpu", False, "use multiple gpu or single gpu")
flags.DEFINE_integer("gpu", 0, "gpu device")
flags.DEFINE_boolean("extend_model", False, "using extended model or not")
flags.DEFINE_boolean("freeze", False, "using extended model or not")
flags.DEFINE_boolean("transfer_all", True, "using extended model or not")
# statistical postfix
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

    model = MultiFeaturesGrafting
    # share model
    use_multiple_gpu = FLAGS.multiple_gpu

    pretrained_master = FullyConv(screen_channels=8, screen_resolution=(
        FLAGS.screen_resolution, FLAGS.screen_resolution))
    pretrained_master.load_state_dict(torch.load(
        './models/collect_five_hierarchical_with_mask_10016017/model_best'
    ))

    # pretrained_sub = FullyConv(screen_channels=8, screen_resolution=(
    #     FLAGS.screen_resolution, FLAGS.screen_resolution))
    # pretrained_sub.load_state_dict(torch.load(
    #     './models/task1_300s_original_16347668/model_best'
    # ))

    long_term_model = FullyConv(screen_channels=8, screen_resolution=(
        FLAGS.screen_resolution, FLAGS.screen_resolution))
    long_term_model.load_state_dict(torch.load(
        './models/task1_insert_no_op_steps_6_3070000/model_latest'
    ))


    if use_multiple_gpu:
        shared_model = model(screen_channels=8, screen_resolution=(
            FLAGS.screen_resolution, FLAGS.screen_resolution))
    else:
        shared_model = model(screen_channels=8, screen_resolution=(
            FLAGS.screen_resolution, FLAGS.screen_resolution)).cuda(FLAGS.gpu)

    if FLAGS.transfer_all:
        shared_model.select_conv1.load_state_dict(pretrained_master.conv1.state_dict())
        shared_model.select_conv2.load_state_dict(pretrained_master.conv2.state_dict())
        shared_model.collect_conv1.load_state_dict(long_term_model.conv1.state_dict())
        shared_model.collect_conv2.load_state_dict(long_term_model.conv2.state_dict())
    else:
        shared_model.collect_conv1.load_state_dict(long_term_model.conv1.state_dict())
        shared_model.collect_conv2.load_state_dict(long_term_model.conv2.state_dict())


    if FLAGS.freeze:
        freeze_layers(shared_model.select_conv1)
        freeze_layers(shared_model.select_conv2)
        freeze_layers(shared_model.collect_conv1)
        freeze_layers(shared_model.collect_conv2)
    else:
        freeze_layers(shared_model.collect_conv1)
        freeze_layers(shared_model.collect_conv2)



    shared_model.share_memory()
    optimizer = SharedAdam(filter(lambda p: p.requires_grad, shared_model.parameters()), lr=FLAGS.learning_rate)
    optimizer.share_memory()

    worker_list = []
    evaluate_worker = mp.Process(target=evaluator, args=(summary_queue, shared_model, optimizer,
                                                         global_counter,
                                                         FLAGS.max_steps,
                                                         FLAGS.postfix))
    evaluate_worker.start()
    worker_list.append(evaluate_worker)

    training_func = train_conjunction_with_action_features
    for worker_id in range(FLAGS.num_of_workers):
        worker = mp.Process(target=training_func,
                            args=(worker_id, FLAGS.flag_values_dict(), shared_model,
                                  optimizer, global_counter, summary_queue))
        worker.start()
        worker_list.append(worker)
    for worker in worker_list:
        worker.join()


if __name__ == '__main__':
    app.run(main)