import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy import stats

trained = [0.20685418, 0.119404815, 0.17970228558406234, 0.20277326, 0.6116525]
untrained = [0.22596654, 0.0050965585, 0.009741321, 0.1028433, 0.21243343]

# Build the plot
fig, ax = plt.subplots()
ax.bar(np.arange(2), [np.mean(trained), np.mean(untrained)], yerr=[sem(trained), sem(untrained)], align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('mean reward across test episodes')
ax.set_xticks(np.arange(2))
ax.set_xticklabels(['trained', 'untrained (random)'])
ax.set_title('Tutor agent\'s performance in transfer to human user')
ax.yaxis.grid(True)

print(stats.ttest_ind(trained, untrained, alternative='greater'))

# Save the figure and show
plt.tight_layout()
# plt.savefig('bar_plot_with_error_bars.png')
plt.show()
