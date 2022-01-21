import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy import stats
import pickle
import pandas as pd

trained = [0.20685418, 0.119404815, 0.17970228558406234, 0.20277326, 0.6116525]
untrained = [0.22596654, 0.0050965585, 0.009741321, 0.1028433, 0.21243343]


def barplot():
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


def reward_plot():
    data = pickle.load(open('data/training_outputs.pickle', 'rb'))[0]
    plt.title('moving average of reward history')
    df = pd.DataFrame(data, columns=['reward'])
    # df['reward'].plot(figsize=(10,6))
    df['reward'].rolling(window=8000).mean().plot()
    plt.show()


reward_plot()



'''
Q: What are constant objects in C++?
A: They're like variables, but you can only assign values to them once. Attempting to do that later throws exceptions.
Q: How can this be applied?
A: If you're working as a programmer in a large codebase, then you can use constant objects to make sure immutable objects are not tampered with.
Q: What's an example of this?
A: An example of using constants might be setting the value of pi, or setting some flags properly which guide behavior later on.
Q: What would be an analogy for this?
A: An analogy to constants might be solid objects, hard materials like diamonds which can't change, and tit's this property precisely which makes them useful.
Q: What needs to be known to judge this?
A: In order to check whether this analogy makes sense, you'd probably need to know a bit more about materials and their use in practice.
Q: What are constant objects in C++?
A: They're like variables, but you can only assign values to them once. Attempting to do that later throws exceptions.
Q: How can this be applied?
A: If you're working as a programmer in a large codebase, then you can use constant objects to make sure immutable objects are not tampered with.
Q: What's an example of this?
A: An example of using constants might be setting the value of pi, or setting some flags properly which guide behavior later on.
Q: What would be an analogy for this?
A: An analogy to constants might be solid objects, hard materials like diamonds which can't change, and tit's this property precisely which makes them useful.
Q: What needs to be known to judge this?
A: In order to check whether this analogy makes sense, you'd probably need to know a bit more about materials and their use in practice.
'''