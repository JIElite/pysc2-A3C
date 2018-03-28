"""
NOTICE: We use current_step > evaluation_target_step and save_statistics_step
to detect the timing for evaluation. By doing so, in main.py the settings for
total game steps need to be little larger than evaluation timing step.
"""
import pickle
import torch



def evaluator(shared_model, global_counter, global_eps, global_avg_perf, global_recently_best_avg):
    statistic_data = {
        'avg_perf': [],
        'recently_best_avg': [],
    }
    recently_best_avg = 0.0
    evaluation_target_step = 10000
    save_statistics_step = 500000

    while True:
        # Anyway, we always save the latest model
        torch.save(shared_model.state_dict(), './models/model_latest')

        # evaluation model by frequency of 10000 steps
        current_step = global_counter.value
        if current_step > evaluation_target_step:

            current_eps = global_eps.value
            average_performance = global_avg_perf.value
            new_recently_best_avg = global_recently_best_avg.value

            print('Elapsed steps:{}, episode: {}, current mean: {}, recently best avg: {}'.format(
                current_step, current_eps, average_performance, new_recently_best_avg
            ))
            print('-------------------------------------------')


            if new_recently_best_avg > recently_best_avg:
                torch.save(shared_model.state_dict(), './models/model_best')
                recently_best_avg = new_recently_best_avg

            # update statistics data
            statistic_data['avg_perf'].append(average_performance)
            statistic_data['recently_best_avg'].append(recently_best_avg)
            evaluation_target_step += 10000

        if current_step > save_statistics_step:
            # write out the statistics
            with open('statistics.pkl', 'wb') as fout:
                pickle.dump(statistic_data, fout)
            save_statistics_step += 500000


def process_statistics(summary_queue, global_eps, global_avg_perf, global_recently_best_avg):
    num_of_eps = 0
    sum_of_eps_return = 0.0
    recently_best_avg = 0.0

    while True:
        frames, episode_reward = summary_queue.get()
        sum_of_eps_return += episode_reward
        num_of_eps += 1
        print('frames: {}, eps score: {}'.format(frames, episode_reward))

        avg_perf = sum_of_eps_return / num_of_eps
        if avg_perf > recently_best_avg:
            recently_best_avg = avg_perf

        global_eps.value = num_of_eps
        global_avg_perf.value = avg_perf
        global_recently_best_avg.value = recently_best_avg


