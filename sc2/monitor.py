import time
from collections import deque

import torch


def evaluator(shared_model, summary_queue, global_counter):
    evaluator_queue = deque(maxlen=100)
    max_recently_score = 0

    while True:
        frames, episode_reward = summary_queue.get()
        print('frames: {}, eps score: {}'.format(frames, episode_reward))

        evaluator_queue.append(episode_reward)
        evaluator_queue_is_full = len(evaluator_queue) == 100

        if evaluator_queue_is_full:
            # start evaluate
            average_recently_performace = sum(evaluator_queue) / 100
            torch.save(shared_model.state_dict(), './models/model_latest'.format(average_recently_performace))

            if global_counter.value != 0 and global_counter.value % 200000 == 0:
                torch.save(shared_model.state_dict(), 'models/model_{}_{}'.format(global_counter.value, average_recently_performace))

            if average_recently_performace > max_recently_score:
                max_recently_score = average_recently_performace
                # save model
                torch.save(shared_model.state_dict(), './models/model_best'.format(max_recently_score))

            print("current mean: {}, recently max: {}".format(average_recently_performace, max_recently_score))

        print('-------------------------------------------')

