import sys
import os
os.environ['OMP_NUM_THREADS']='1'

from absl import app, flags
from pysc2 import maps
from pysc2.env import sc2_env
import torch
import torch.multiprocessing as mp

from model import Grafting_MultiunitCollect, ExtendConv3Grafting_MultiunitCollect, FullyConv
from optimizer import SharedAdam
from monitor import evaluator

# TODO designed new training procedure
from train_hierarchical import train_conjunction
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
flags.DEFINE_integer('worker_steps', 10500000, "steps run for each worker")
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

    if FLAGS.extend_model:
        model = ExtendConv3Grafting_MultiunitCollect
    else:
        model = Grafting_MultiunitCollect

    # share model
    use_multiple_gpu = FLAGS.multiple_gpu

    pretrained_master = FullyConv(screen_channels=8, screen_resolution=(
        FLAGS.screen_resolution, FLAGS.screen_resolution))
    pretrained_master.load_state_dict(torch.load(
        './models/collect_five_hierarchical_with_mask_10016017/model_best'
    ))

    pretrained_sub = FullyConv(screen_channels=8, screen_resolution=(
        FLAGS.screen_resolution, FLAGS.screen_resolution))
    pretrained_sub.load_state_dict(torch.load(
        './models/task1_300s_original_16347668/model_best'
    ))


    if use_multiple_gpu:
        shared_model = model(screen_channels=8, screen_resolution=(
            FLAGS.screen_resolution, FLAGS.screen_resolution))
    else:
        shared_model = model(screen_channels=8, screen_resolution=(
            FLAGS.screen_resolution, FLAGS.screen_resolution)).cuda(FLAGS.gpu)

    shared_model.conv_master.load_state_dict(pretrained_master.conv1.state_dict())
    shared_model.conv_sub.load_state_dict(pretrained_sub.conv1.state_dict())
    shared_model.spatial_policy.load_state_dict(pretrained_sub.spatial_policy.state_dict())
    shared_model.select_unit.load_state_dict(pretrained_master.spatial_policy.state_dict())
    # shared_model.non_spatial_branch.load_state_dict(pretrained_master.non_spatial_branch.state_dict())
    # shared_model.value.load_state_dict(pretrained_master.value.state_dict())

    freeze_layers(shared_model.conv_master)
    freeze_layers(shared_model.conv_sub)
    freeze_layers(shared_model.spatial_policy)
    freeze_layers(shared_model.select_unit)
    # freeze_layers(shared_model.non_spatial_branch)
    # freeze_layers(shared_model.value)


    shared_model.share_memory()
    optimizer = SharedAdam(filter(lambda p: p.requires_grad, shared_model.parameters()), lr=FLAGS.learning_rate)
    optimizer.share_memory()

    worker_list = []
    evaluate_worker = mp.Process(target=evaluator, args=(summary_queue, shared_model, optimizer,
                                                         global_counter,
                                                         FLAGS.num_of_workers * FLAGS.worker_steps,
                                                         FLAGS.postfix))
    evaluate_worker.start()
    worker_list.append(evaluate_worker)

    for worker_id in range(FLAGS.num_of_workers):
        worker = mp.Process(target=train_conjunction,
                            args=(worker_id, FLAGS.flag_values_dict(), shared_model,
                                  optimizer, global_counter, summary_queue))
        worker.start()
        worker_list.append(worker)
    for worker in worker_list:
        worker.join()


if __name__ == '__main__':
    app.run(main)