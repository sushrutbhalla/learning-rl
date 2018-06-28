import numpy as np
import MDP
import RL2
from Utils2 import print_policy_word, check_stochastic_policy_terminate, generate_data_for_plot, plot_avg_cumulative_reward


''' Construct a simple maze MDP

  Grid world layout:

  ---------------------
  |  0 |  1 |  2 |  3 |
  ---------------------
  |  4 |  5 |  6 |  7 |
  ---------------------
  |  8 |  9 | 10 | 11 |
  ---------------------
  | 12 | 13 | 14 | 15 |
  ---------------------

  Goal state: 15
  Bad state: 9
  End state: 16

  The end state is an absorbing state that the agent transitions
  to after visiting the goal state.

  There are 17 states in total (including the end state)
  and 4 actions (up, down, left, right).'''

# Transition function: |A| x |S| x |S'| array
T = np.zeros([4,17,17])
a = 0.8;  # intended move
b = 0.1;  # lateral move

# up (a = 0)

T[0,0,0] = a+b;
T[0,0,1] = b;

T[0,1,0] = b;
T[0,1,1] = a;
T[0,1,2] = b;

T[0,2,1] = b;
T[0,2,2] = a;
T[0,2,3] = b;

T[0,3,2] = b;
T[0,3,3] = a+b;

T[0,4,4] = b;
T[0,4,0] = a;
T[0,4,5] = b;

T[0,5,4] = b;
T[0,5,1] = a;
T[0,5,6] = b;

T[0,6,5] = b;
T[0,6,2] = a;
T[0,6,7] = b;

T[0,7,6] = b;
T[0,7,3] = a;
T[0,7,7] = b;

T[0,8,8] = b;
T[0,8,4] = a;
T[0,8,9] = b;

T[0,9,8] = b;
T[0,9,5] = a;
T[0,9,10] = b;

T[0,10,9] = b;
T[0,10,6] = a;
T[0,10,11] = b;

T[0,11,10] = b;
T[0,11,7] = a;
T[0,11,11] = b;

T[0,12,12] = b;
T[0,12,8] = a;
T[0,12,13] = b;

T[0,13,12] = b;
T[0,13,9] = a;
T[0,13,14] = b;

T[0,14,13] = b;
T[0,14,10] = a;
T[0,14,15] = b;

T[0,15,16] = 1;
T[0,16,16] = 1;

# down (a = 1)

T[1,0,0] = b;
T[1,0,4] = a;
T[1,0,1] = b;

T[1,1,0] = b;
T[1,1,5] = a;
T[1,1,2] = b;

T[1,2,1] = b;
T[1,2,6] = a;
T[1,2,3] = b;

T[1,3,2] = b;
T[1,3,7] = a;
T[1,3,3] = b;

T[1,4,4] = b;
T[1,4,8] = a;
T[1,4,5] = b;

T[1,5,4] = b;
T[1,5,9] = a;
T[1,5,6] = b;

T[1,6,5] = b;
T[1,6,10] = a;
T[1,6,7] = b;

T[1,7,6] = b;
T[1,7,11] = a;
T[1,7,7] = b;

T[1,8,8] = b;
T[1,8,12] = a;
T[1,8,9] = b;

T[1,9,8] = b;
T[1,9,13] = a;
T[1,9,10] = b;

T[1,10,9] = b;
T[1,10,14] = a;
T[1,10,11] = b;

T[1,11,10] = b;
T[1,11,15] = a;
T[1,11,11] = b;

T[1,12,12] = a+b;
T[1,12,13] = b;

T[1,13,12] = b;
T[1,13,13] = a;
T[1,13,14] = b;

T[1,14,13] = b;
T[1,14,14] = a;
T[1,14,15] = b;

T[1,15,16] = 1;
T[1,16,16] = 1;

# left (a = 2)

T[2,0,0] = a+b;
T[2,0,4] = b;

T[2,1,1] = b;
T[2,1,0] = a;
T[2,1,5] = b;

T[2,2,2] = b;
T[2,2,1] = a;
T[2,2,6] = b;

T[2,3,3] = b;
T[2,3,2] = a;
T[2,3,7] = b;

T[2,4,0] = b;
T[2,4,4] = a;
T[2,4,8] = b;

T[2,5,1] = b;
T[2,5,4] = a;
T[2,5,9] = b;

T[2,6,2] = b;
T[2,6,5] = a;
T[2,6,10] = b;

T[2,7,3] = b;
T[2,7,6] = a;
T[2,7,11] = b;

T[2,8,4] = b;
T[2,8,8] = a;
T[2,8,12] = b;

T[2,9,5] = b;
T[2,9,8] = a;
T[2,9,13] = b;

T[2,10,6] = b;
T[2,10,9] = a;
T[2,10,14] = b;

T[2,11,7] = b;
T[2,11,10] = a;
T[2,11,15] = b;

T[2,12,8] = b;
T[2,12,12] = a+b;

T[2,13,9] = b;
T[2,13,12] = a;
T[2,13,13] = b;

T[2,14,10] = b;
T[2,14,13] = a;
T[2,14,14] = b;

T[2,15,16] = 1;
T[2,16,16] = 1;

# right (a = 3)

T[3,0,0] = b;
T[3,0,1] = a;
T[3,0,4] = b;

T[3,1,1] = b;
T[3,1,2] = a;
T[3,1,5] = b;

T[3,2,2] = b;
T[3,2,3] = a;
T[3,2,6] = b;

T[3,3,3] = a+b;
T[3,3,7] = b;

T[3,4,0] = b;
T[3,4,5] = a;
T[3,4,8] = b;

T[3,5,1] = b;
T[3,5,6] = a;
T[3,5,9] = b;

T[3,6,2] = b;
T[3,6,7] = a;
T[3,6,10] = b;

T[3,7,3] = b;
T[3,7,7] = a;
T[3,7,11] = b;

T[3,8,4] = b;
T[3,8,9] = a;
T[3,8,12] = b;

T[3,9,5] = b;
T[3,9,10] = a;
T[3,9,13] = b;

T[3,10,6] = b;
T[3,10,11] = a;
T[3,10,14] = b;

T[3,11,7] = b;
T[3,11,11] = a;
T[3,11,15] = b;

T[3,12,8] = b;
T[3,12,13] = a;
T[3,12,12] = b;

T[3,13,9] = b;
T[3,13,14] = a;
T[3,13,13] = b;

T[3,14,10] = b;
T[3,14,15] = a;
T[3,14,14] = b;

T[3,15,16] = 1;
T[3,16,16] = 1;

# Reward function: |A| x |S| array
R = -1 * np.ones([4,17]);

# set rewards
R[:,15] = 100;  # goal state
R[:,9] = -70;   # bad state
R[:,16] = 0;    # end state

# Discount factor: scalar in [0,1)
discount = 0.95

# MDP object
mdp = MDP.MDP(T,R,discount)

# RL problem
rlProblem = RL2.RL2(mdp,np.random.normal)

# Test REINFORCE
Q, policy = rlProblem.reinforce(s0=0,initialPolicyParams=np.random.rand(mdp.nActions,mdp.nStates),nEpisodes=200,nSteps=100)
print ("\nREINFORCE results")
print (policy)
print ("rounded policy: \n{}".format(np.round(policy, decimals=1)))
argmax_policy = np.argmax(policy, axis=0)
print ("argmax policy: \n{}".format(argmax_policy))
print_policy_word(argmax_policy, rlProblem, s0=0, nSteps=100)
#Note the previous function call only checks if the argmax policy will terminate
#Check over 100 trials if current stochastic policy will terminate
check_stochastic_policy_terminate(policy, rlProblem, s0=0, nSteps=100, nTrials=100)
print ("last 10 episode rewards: {}".format(rlProblem.get_reinforce_cumulative_reward()[-10:]))
#mean episode reward doesn't mean much
# print ("mean episode reward: {}".format(np.mean(rlProblem.get_reinforce_cumulative_reward())))

# Test model-based RL
[V,policy] = rlProblem.modelBasedRL(s0=0,defaultT=np.ones([mdp.nActions,mdp.nStates,mdp.nStates])/mdp.nStates,initialR=np.zeros([mdp.nActions,mdp.nStates]),nEpisodes=200,nSteps=100,epsilon=0.3)
print ("\nmodel-based RL results")
print (V)
print (policy)
print ("rounded policy: \n{}".format(np.round(policy, decimals=1)))
print_policy_word(policy, rlProblem, s0=0, nSteps=100)

# Test Q-learning
[Q,policy] = rlProblem.qLearning(s0=0,initialQ=np.zeros([mdp.nActions,mdp.nStates]),nEpisodes=200,nSteps=100,epsilon=0.05)
print ("\nQ-learning results")
print (Q)
print (policy)
print_policy_word(policy, rlProblem, s0=0, nSteps=100)


############################################################
#plot results for REINFORCE, ModelBasedRL and qLearning

#generate data for 200 episodes over 100 trials
rlProblem_results = generate_data_for_plot(rlProblem)
plot_legend = rlProblem_results.pop(-1)
plot_filename = "results/maze_avg_cumulative_reward_part1.png"
plot_title = "Maze: Original Average Cumulative Reward"
plot_avg_cumulative_reward(rlProblem_results, plot_legend, plot_title, plot_filename, use_ax_limit=False)

#plot smoothed curves
plot_filename = "results/maze_smooth_n5_avg_cumulative_reward_part1.png"
plot_title = "Maze: Smooth(n=5) Average Cumulative Reward"
plot_avg_cumulative_reward(rlProblem_results, plot_legend, plot_title, plot_filename, n=5, smooth=True, use_ax_limit=False)
plot_filename = "results/maze_smooth_n10_avg_cumulative_reward_part1.png"
plot_title = "Maze: Smooth(n=10) Average Cumulative Reward"
plot_avg_cumulative_reward(rlProblem_results, plot_legend, plot_title, plot_filename, n=10, smooth=True, use_ax_limit=False)
plot_filename = "results/maze_smooth_n20_avg_cumulative_reward_part1.png"
plot_title = "Maze: Smooth(n=20) Average Cumulative Reward"
plot_avg_cumulative_reward(rlProblem_results, plot_legend, plot_title, plot_filename, n=20, smooth=True, use_ax_limit=False)
plot_filename = "results/maze_smooth_n50_avg_cumulative_reward_part1.png"
plot_title = "Maze: Smooth(n=50) Average Cumulative Reward"
plot_avg_cumulative_reward(rlProblem_results, plot_legend, plot_title, plot_filename, n=50, smooth=True, use_ax_limit=False)
