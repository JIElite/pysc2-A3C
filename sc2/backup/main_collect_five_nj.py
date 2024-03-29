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
    Grafting_MultiunitCollect,
    ExtendConv3Grafting_MultiunitCollect,
    Grafting_MultiunitCollect_WithActionFeatures,
    ExtendConv3Grafting_MultiunitCollect_WithActionFeatures,
    MultiInputSinglePolicyNet,
    MultiInputSinglePolicyNetExtendConv3,
)
from model2 import CollectFiveDropout, CollectFiveDropoutConv3, CollectFiveDropoutConv4
from optimizer import SharedAdam
from monitor import evaluator

from train_hierarchical import train_conjunction
from train_collect_five_nj import train_conjunction_with_action_features
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
flags.DEFINE_integer("version", 2, "model type")
flags.DEFINE_boolean("short_term", False, "long-term collection policy or short-term")
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


    if FLAGS.version == 1:
        if FLAGS.extend_model:
            model = ExtendConv3Grafting_MultiunitCollect
        else:
            model = Grafting_MultiunitCollect
    elif FLAGS.version == 2:
        if FLAGS.extend_model:
            model = ExtendConv3Grafting_MultiunitCollect_WithActionFeatures
        else:
            model = Grafting_MultiunitCollect_WithActionFeatures
    elif FLAGS.version == 3:
        if FLAGS.extend_model:
            model = MultiInputSinglePolicyNetExtendConv3
        else:
            model = MultiInputSinglePolicyNet
    elif FLAGS.version == 4:
        # dropout net
        if FLAGS.extend_model:
            # model = CollectFiveDropoutConv3
            model = CollectFiveDropoutConv4
        else:
            model = CollectFiveDropout

    print("Model type:", model)
    # share model
    use_multiple_gpu = FLAGS.multiple_gpu

    pretrained_master = FullyConv(screen_channels=8, screen_resolution=(
        FLAGS.screen_resolution, FLAGS.screen_resolution))
    pretrained_master.load_state_dict(torch.load(
        './models/collect_five_hierarchical_with_mask_10016017/model_best'
    ))

    if FLAGS.short_term:
        pretrained_sub = FullyConv(screen_channels=8, screen_resolution=(
            FLAGS.screen_resolution, FLAGS.screen_resolution))
        pretrained_sub.load_state_dict(torch.load(
            './models/task1_300s_original_16347668/model_best'
        ))
    else:
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

    shared_model.train()
    shared_model.conv_master.load_state_dict(pretrained_master.conv1.state_dict())
    shared_model.select_unit.load_state_dict(pretrained_master.spatial_policy.state_dict())

    if FLAGS.short_term:
        shared_model.conv_sub.load_state_dict(pretrained_sub.conv1.state_dict())
        shared_model.spatial_policy.load_state_dict(pretrained_sub.spatial_policy.state_dict())
    else:
        shared_model.conv_sub.load_state_dict(long_term_model.conv1.state_dict())
        shared_model.spatial_policy.load_state_dict(long_term_model.spatial_policy.state_dict())

    # freeze_layers(shared_model.conv_master)
    # freeze_layers(shared_model.conv_sub)
    # freeze_layers(shared_model.spatial_policy)
    # freeze_layers(shared_model.select_unit)


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

    # training
    # training_func = train_conjunction
    # if FLAGS.with_action:
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