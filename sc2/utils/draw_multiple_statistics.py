"""
usage:
python show_statistics.py --input statistics/collect_five_nj_extend_conv3_14045348.pkl --steps 400000
"""
import pickle
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--input', dest='source', default='statistics.pkl', help='input of statistics file')
parser.add_argument('--steps', dest='steps', default=None, help='display to num_of_steps')
args = parser.parse_args()

with open('./neural_network_comparison/collect_five_sparse_positive_reward/without conv3/transfer_short_term_policy_lr5e-4/statistics_our_short_6.pkl', 'rb') as fd, \
    open('neural_network_comparison/collect_five_sparse_positive_reward/without conv3/transfer_short_term_policy_lr5e-4/statistics_our_short_1.pkl', 'rb') as fd2, \
    open('neural_network_comparison/collect_five_sparse_positive_reward/without conv3/transfer_short_term_policy_lr5e-4/statistics_our_short_2.pkl', 'rb') as fd3:
    # open('./collect_five_new_nj_worker12_freeze_policy_extend_conv3/statistics/ten_run.pkl', 'rb') as fd4:
        data = pickle.load(fd)
        data2 = pickle.load(fd2)
        data3 = pickle.load(fd3)
        # data4 = pickle.load(fd4)

if args.steps is None:
    end_of_steps = len(data['recent_100_mean_perf'])
else:
    end_of_steps = int(args.steps)

plt.plot(data['recent_100_mean_perf'][:end_of_steps], label='short-term policy1')
plt.plot(data2['recent_100_mean_perf'][:end_of_steps], label='short-term policy2')
plt.plot(data3['recent_100_mean_perf'][:end_of_steps], label='short-term policy3')
# plt.plot(data4['recent_100_mean_perf'][:end_of_steps], label='grafted extend 1 conv (10 runs)')
plt.legend(loc='lower right')
plt.ylabel('average episodic rewards')
plt.xlabel('steps (1e4)')
plt.show()