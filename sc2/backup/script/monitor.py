from collections import deque



def evaluator(summary_queue):
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

            if average_recently_performace > max_recently_score:
                max_recently_score = average_recently_performace

            print("current mean: {}, recently max: {}".format(average_recently_performace, max_recently_score))

        print('-------------------------------------------')