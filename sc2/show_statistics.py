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


with open(args.source, 'rb') as fd:
    data = pickle.load(fd)

if args.steps is None:
    end_of_steps = len(data['mean_perf'])
else:
    end_of_steps = int(int(args.steps) / 10000)

plt.plot(data['mean_perf'][:end_of_steps])
plt.plot(data['recent_100_mean_perf'][:end_of_steps])
plt.plot(data['recent_100_best_mean_perf'][:end_of_steps])
plt.ylabel('average return')
plt.xlabel('steps (1e4)')
plt.show()