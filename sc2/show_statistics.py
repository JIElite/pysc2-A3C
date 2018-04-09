import pickle
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--input', dest='source', default='statistics.pkl', help='input of statistics file')
args = parser.parse_args()

with open(args.source, 'rb') as fd:
    data = pickle.load(fd)

plt.plot(data['mean_perf'])
plt.plot(data['recent_100_mean_perf'])
plt.plot(data['recent_100_best_mean_perf'])
plt.ylabel('average return')
plt.xlabel('steps (1e4)')
plt.show()