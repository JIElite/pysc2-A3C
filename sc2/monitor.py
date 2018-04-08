"""
NOTICE: We use current_step > evaluation_target_step and save_statistics_step
to detect the timing for evaluation. By doing so, in main.py the settings for
total game steps need to be little larger than evaluation timing step.
"""
from collections import deque

import pickle
import torch


def evaluator(summary_queue, shared_model, global_counter, end_of_steps=1000000):
    '''
    Emit avg_perf and recently_best avg
    '''
    statistics = {
        'mean_perf': [],
        'best_mean_perf': [],
        'recent_100_mean_perf': [],
        'recent_100_best_mean_perf': []
    }

    num_of_eps = 0
    sum_of_eps_return = 0.0
    best_mean_perf = 0.0
    recent_perf_queue = deque(maxlen=100)
    recent_100_best_mean_perf = 0.0

    evaluation_steps = 10000

    while True:
        if global_counter.value >= end_of_steps - 3000:
            torch.save(shared_model.state_dict(), './models/model_latest')
            with open('statistics.pkl', 'wb') as fout:
                pickle.dump(statistics, fout)
            return

        frames, episode_reward = summary_queue.get()
        sum_of_eps_return += episode_reward
        num_of_eps += 1
        recent_perf_queue.append(episode_reward)
        print('frames: {}, eps score: {}'.format(frames, episode_reward))

        # evaluate recently performance per 10000 frames
        current_step = global_counter.value
        if len(recent_perf_queue) > 20 and current_step > evaluation_steps:
            # update latest model
            torch.save(shared_model.state_dict(), './models/model_latest')

            # compute mean perf
            recent_100_mean_perf = sum(recent_perf_queue) / len(recent_perf_queue)
            mean_perf = sum_of_eps_return / num_of_eps

            if recent_100_mean_perf > recent_100_best_mean_perf:
                recent_100_best_mean_perf = recent_100_mean_perf
                torch.save(shared_model.state_dict(), './models/model_best')

            if mean_perf > best_mean_perf:
                best_mean_perf = mean_perf

            print('Elapsed steps:{}, episode: {}, mean: {}, best mean: {}, recent 100 mean: {}, recent 100 best mean: {}'\
                .format(current_step, num_of_eps, mean_perf, best_mean_perf, recent_100_mean_perf, recent_100_best_mean_perf))
            print('-------------------------------------------')

            statistics['mean_perf'].append(mean_perf)
            statistics['best_mean_perf'].append(best_mean_perf)
            statistics['recent_100_mean_perf'].append(recent_100_mean_perf)
            statistics['recent_100_best_mean_perf'].append(recent_100_best_mean_perf)
            evaluation_steps += 10000


        # store statistics per 10000 episodes
        if num_of_eps % 500 == 0:
            with open('statistics.pkl', 'wb') as fout:
                pickle.dump(statistics, fout)


