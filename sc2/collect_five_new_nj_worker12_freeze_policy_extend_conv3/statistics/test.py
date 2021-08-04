import matplotlib.pyplot as plt
import pickle


statistics = []
for i in range(2, 11):
    with open('./{}.pkl'.format(i), 'rb') as fd:
        data = pickle.load(fd)
        statistics.append(data['recent_100_mean_perf'])

avg_result = []
for records in zip(*statistics):
    avg_result.append(sum(records) / len(records))

ten_run_statistics = {
    'recent_100_mean_perf': avg_result
}

with open('./ten_run_statistics.pkl', 'wb') as fd:
    pickle.dump(ten_run_statistics, fd)

plt.plot(avg_result)
plt.show()

