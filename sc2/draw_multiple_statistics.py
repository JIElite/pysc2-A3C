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


with open('s1.pkl', 'rb') as fd, open('s2.pkl', 'rb') as fd2, \
    open('./statistics/collect_five_new_nj_worker12_freeze_policy.pkl', 'rb') as fd3, \
    open('./statistics/collect_five_new_nj_worker12_finetune_policy.pkl', 'rb') as fd4, \
    open('./statistics/collect_five_new_nj_worker12_freeze_policy_2.pkl', 'rb') as fd5:
    data = pickle.load(fd)
    data2 = pickle.load(fd2)
    data3 = pickle.load(fd3)
    data4 = pickle.load(fd4)
    data5 = pickle.load(fd5)

end_of_steps = 300
plt.plot(data['recent_100_mean_perf'][:end_of_steps], label='from scratch')
plt.plot(data2['recent_100_mean_perf'][:end_of_steps], label='new finetune')
plt.plot(data3['recent_100_mean_perf'][:end_of_steps], label='freeze') 
plt.plot(data4['recent_100_mean_perf'][:end_of_steps], label='finetune')
plt.plot(data5['recent_100_mean_perf'][:end_of_steps], label='freeze2')
plt.legend(loc='lower right')
plt.ylabel('average return')
plt.xlabel('steps (1e4)')
plt.show()