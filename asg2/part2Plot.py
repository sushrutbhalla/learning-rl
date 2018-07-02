from __future__ import print_function, division
import numpy as np
from Utils2 import generate_bandit_data_for_plot, plot_avg_cumulative_reward

#read result file
dir_name = 'asg2_ddpg_logs_no_noise_in_if'
episode_reward_normal_file = 'results/{}/no_sharing/history_episode_reward_normal.csv'.format('asg2_ddpg_logs')
episode_reward_share_top_layer_file = 'results/{}/sharing/history_episode_reward_share_top_layer.csv'.format(dir_name)

episode_reward_normal = np.genfromtxt(episode_reward_normal_file, delimiter=',')
episode_reward_share_top_layer = np.genfromtxt(episode_reward_share_top_layer_file, delimiter=',')
plot_legend = []
plot_legend.append('Regular DDPG')
plot_legend.append('Shared layer DDPG')

print ("episode_reward_normal.shape: {}".format(episode_reward_normal.shape))
print ("episode_reward_share_top_layer.shape: {}".format(episode_reward_share_top_layer.shape))

############################################################
#plot results for DDPG and shared layers DDPG

plot_filename = "results/{}/ddpg_part2.png".format(dir_name)
plot_title = "DDPG: Original Cumulative Reward"
plot_avg_cumulative_reward([episode_reward_normal, episode_reward_share_top_layer], \
  plot_legend, plot_title, plot_filename, avg_rew=False, use_ax_limit=False)

#plot smoothed curves
plot_filename = "results/{}/ddpg_smooth_n5_part2.png".format(dir_name)
plot_title = "DDPG: Smooth(n=5) Cumulative Reward"
plot_avg_cumulative_reward([episode_reward_normal, episode_reward_share_top_layer], \
  plot_legend, plot_title, plot_filename, n=5, avg_rew=False, smooth=True, use_ax_limit=False)
plot_filename = "results/{}/ddpg_smooth_n10_part2.png".format(dir_name)
plot_title = "DDPG: Smooth(n=10) Cumulative Reward"
plot_avg_cumulative_reward([episode_reward_normal, episode_reward_share_top_layer], \
  plot_legend, plot_title, plot_filename, n=10, avg_rew=False, smooth=True, use_ax_limit=False)
plot_filename = "results/{}/ddpg_smooth_n20_part2.png".format(dir_name)
plot_title = "DDPG: Smooth(n=20) Cumulative Reward"
plot_avg_cumulative_reward([episode_reward_normal, episode_reward_share_top_layer], \
  plot_legend, plot_title, plot_filename, n=20, avg_rew=False, smooth=True, use_ax_limit=False)
plot_filename = "results/{}/ddpg_smooth_n50_part2.png".format(dir_name)
plot_title = "DDPG: Smooth(n=50) Cumulative Reward"
plot_avg_cumulative_reward([episode_reward_normal, episode_reward_share_top_layer], \
  plot_legend, plot_title, plot_filename, n=50, avg_rew=False, smooth=True, use_ax_limit=False)
