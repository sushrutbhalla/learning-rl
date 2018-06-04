from __future__ import division
import numpy as np
from custom_cartpole import get_cartpole_rewards
from Utils import plot_avg_cumulative_reward

nEpisodes = 1000

#generate plots for different number of steps before target network update
reward_target_upd_step = []
episodes_to_solve_upd_steps = np.zeros([4], dtype=int)
plot_legend = []
plot_title = "DQN: Cumulative reward per episode in CartPole Env"
plot_filename = "results/cartpole_target_upd_step.png"
for idx,steps in enumerate(list([1000, 250, 50, 1])):
    cumulative_reward,episodes_to_solve_upd_steps[idx] = get_cartpole_rewards(target_net_upd_steps=steps,nEpisodes=nEpisodes)
    reward_target_upd_step.append(cumulative_reward)
    plot_legend.append('update steps: {}'.format(steps))
    print ("[DEBUG] to_solve: {}, done: {}".format(episodes_to_solve_upd_steps[idx], steps))
print ("Episodes to Solve UPD STEPS: {}".format(episodes_to_solve_upd_steps))
plot_avg_cumulative_reward(reward_target_upd_step, plot_legend, plot_title, plot_filename)
plot_filename = "results/cartpole_target_upd_step_smooth_5.png"
plot_avg_cumulative_reward(reward_target_upd_step, plot_legend, plot_title, plot_filename, smooth=True, n=5)
plot_filename = "results/cartpole_target_upd_step_smooth_10.png"
plot_avg_cumulative_reward(reward_target_upd_step, plot_legend, plot_title, plot_filename, smooth=True, n=10)
plot_filename = "results/cartpole_target_upd_step_smooth_20.png"
plot_avg_cumulative_reward(reward_target_upd_step, plot_legend, plot_title, plot_filename, smooth=True, n=20)
plot_filename = "results/cartpole_target_upd_step_smooth_50.png"
plot_avg_cumulative_reward(reward_target_upd_step, plot_legend, plot_title, plot_filename, smooth=True, n=50)

#generate plots for different number of mini-batch size for update using replay buffer
reward_mini_batch_size = []
episodes_to_solve_batch_size = np.zeros([4])
plot_legend = []
plot_title = "DQN: Cumulative reward per episode in CartPole Env"
plot_filename = "results/cartpole_mini_batch_size.png"
for idx,size in enumerate(list([32,15,5,1])):
    cumulative_reward,episodes_to_solve_batch_size[idx] = get_cartpole_rewards(mini_batch_sample_size=size,nEpisodes=nEpisodes)
    reward_mini_batch_size.append(cumulative_reward)
    plot_legend.append('mini-batch size: {}'.format(size))
    print ("[DEBUG] to_solve: {}, done: {}".format(episodes_to_solve_batch_size[idx], size))
print ("Episodes to Solve BATCH SIZE: {}".format(episodes_to_solve_batch_size))
plot_avg_cumulative_reward(reward_mini_batch_size, plot_legend, plot_title, plot_filename)
plot_filename = "results/cartpole_mini_batch_size_smooth_5.png"
plot_avg_cumulative_reward(reward_mini_batch_size, plot_legend, plot_title, plot_filename, smooth=True, n=5)
plot_filename = "results/cartpole_mini_batch_size_smooth_10.png"
plot_avg_cumulative_reward(reward_mini_batch_size, plot_legend, plot_title, plot_filename, smooth=True, n=10)
plot_filename = "results/cartpole_mini_batch_size_smooth_20.png"
plot_avg_cumulative_reward(reward_mini_batch_size, plot_legend, plot_title, plot_filename, smooth=True, n=20)
plot_filename = "results/cartpole_mini_batch_size_smooth_50.png"
plot_avg_cumulative_reward(reward_mini_batch_size, plot_legend, plot_title, plot_filename, smooth=True, n=50)

