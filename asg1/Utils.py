import numpy as np
import matplotlib.pyplot as plt

#def plot_avg_cumulative_reward(avg_cumulative_reward, legend, title, filename):
#    for idx in range(avg_cumulative_reward.shape[0]):
#        plt.plot(avg_cumulative_reward[idx,:])
#    plt.title(title)
#    plt.ylabel('Cumulative Reward')
#    plt.xlabel('Episode')
#    plt.legend(legend, loc='lower right')
#    plt.savefig(filename)
#    plt.show()

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    final = ret[n - 1:] / n
    return final
    #pad = final[-1]*np.ones(len(a)-len(final))
    #return np.concatenate((final,pad),axis=0)

def plot_avg_cumulative_reward(avg_cumulative_reward, legend, title, filename, smooth=False, n=10, ymin=-20, ymax=220):
    for idx in range(len(avg_cumulative_reward)):
        if not smooth:
            plt.plot(avg_cumulative_reward[idx])
        else:
            plt.plot(moving_average(avg_cumulative_reward[idx],n=n))
    plt.title(title)
    if not smooth:
        plt.ylabel('Cumulative Reward')
    else:
        plt.ylabel('Smoothed({}) Cumulative Reward'.format(n))
    plt.xlabel('Episode')
    axes = plt.gca()
    #axes.set_xlim([xmin,xmax])
    axes.set_ylim([ymin,ymax])
    plt.legend(legend, loc='lower right')
    plt.savefig(filename)
    plt.show()



