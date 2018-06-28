import numpy as np
import matplotlib.pyplot as plt

def check_for_goodness(policy):
  ''' based on my preception of the grid, is the policy good?
  policy: current policy to evaluate
  '''
  good_states = []
  bad_states = []
  for state,action in enumerate(policy):
    if state == 9 or state == 15 or state == 16:
      good_states.append(state) #9 wouldn't have converged as all states will try to avoid it so the n(s,a) for state-9 is not close to inifinite
    elif state <= 11 and state != 5 and action == 1:
      good_states.append(state)
    elif state <=14 and state != 8 and state != 3 and state != 7 and state != 11 and action == 3:
      good_states.append(state)
    elif state == 5 and action != 1:
      good_states.append(state)
    elif state == 8 and action != 3: #the transition prob for T[2,8,8]=a which makes it think left will help stay in 8
      good_states.append(state)
    else:
      bad_states.append(state)
  assert len(good_states) + len(bad_states) == len(policy), "Length of the states don't match with policy"
  if len(good_states) == len(policy):
    return True, None
  return False, bad_states

def turn_towards_pit(policy):
  '''soft goodness. any action is OK in the policy as long as the policy doesn't end up in the pit
  policy: current policy to evaluate
  '''
  for state,action in enumerate(policy):
    if state == 5 and action == 1:
      return True
    elif state == 8 and action == 3:
      return True
    elif state == 13 and action == 0:
      return True
    elif state == 10 and action == 2:
      return True
  return False

def policy_reach_terminal(policy, rlProblem, s0=0, nSteps=100):
  '''Does the current policy reach terminal state within one episode?
  policy: current policy to evaluate
  '''
  state = s0
  total_reward = 0
  for idx in range(nSteps):
    action = policy[state]
    reward, state_p = rlProblem.sampleRewardAndNextState(state, action)
    state = state_p
    total_reward += reward
    if state == 16:
      return True, total_reward, idx
  return False, total_reward, nSteps

def print_policy_word(policy, rlProblem, s0, nSteps):
  policy_word = []
  for state,action in enumerate(policy):
    if action == 0:
      policy_word.append(str(state)+':up')
    elif action == 1:
      policy_word.append(str(state)+':down')
    elif action == 2:
      policy_word.append(str(state)+':left')
    elif action == 3:
      policy_word.append(str(state)+':right')
    else:
      print ("[ERROR] wrong action chosen in policy: \n{}".format(policy))
      exit(-1)
  assert len(policy) == len(policy_word), "Length of policy_word doesn't match length of policy"
  goodness, bad_states = check_for_goodness(policy)
  terminate, total_reward, total_steps_terminate = policy_reach_terminal(policy, rlProblem, s0, nSteps)
  print ("TERMINATE: {}, GOODNESS: {}, SOFT_GOODNESS: {}\nbad states: {}, Total Reward: {}, Total Steps to terminate: {}\n{}".format(\
    terminate, goodness, not turn_towards_pit(policy), bad_states,\
    total_reward, total_steps_terminate, policy_word))
  return


def check_stochastic_policy_terminate(policy, rlProblem, s0, nSteps, nTrials):
    '''Does the current policy reach terminal state within one episode?
    policy: current policy to evaluate
    '''
    n_terminate = 0
    for trial in range(nTrials):
        state = s0
        total_reward = 0
        terminate = False
        for idx in range(nSteps):
            action = np.random.choice(policy.shape[0], p=policy[:,state])
            reward, state_p = rlProblem.sampleRewardAndNextState(state, action)
            state = state_p
            total_reward += reward
            if state == 16:
                terminate += 1
                terminate = True
        if terminate:
            n_terminate += 1
    print ("Number of terminations: {}".format(n_terminate))
    return n_terminate

def moving_average(a, n=3, add_pad=False):
    '''Compute the moving average of array a using a window of size n
    add_pad: if True, a padding to the final array is added to make it equal in size to input array
    '''
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    final = ret[n - 1:] / n
    if not add_pad:
        return final
    else:
        pad = final[-1]*np.ones(len(a)-len(final))
        return np.concatenate((final,pad),axis=0)

def plot_avg_cumulative_reward(avg_cumulative_reward, legend, title, filename, avg_rew=True, smooth=False, n=10, use_ax_limit=True, ymin=-20, ymax=220):
    ''' Plot the list of list of avg_cumulative_reward on a single graph.
    Most of the input variable names are fairly intuitive and others are detailed below:
    filename: specify a filename if the plot should be saved
    smooth: generate a smooth graph using moving average of n steps
    n: number of steps to use to generate a moving average
    use_ax_limit: limit the x and y axes values with a maximum and minimum (ymin and ymax)
    '''
    for idx in range(len(avg_cumulative_reward)):
        if not smooth:
            plt.plot(avg_cumulative_reward[idx])
        else:
            plt.plot(moving_average(avg_cumulative_reward[idx],n=n))
    plt.title(title)
    y_plt_label = ''
    if not smooth:
        y_plt_label = 'Cumulative Reward'
    else:
        y_plt_label = 'Smoothed({}) Cumulative Reward'.format(n)
    if avg_rew:
        y_plt_label = 'Avg: ' + y_plt_label
    plt.ylabel(y_plt_label)
    plt.xlabel('Episode')
    if use_ax_limit:
        axes = plt.gca()
        #axes.set_xlim([xmin,xmax])
        axes.set_ylim([ymin,ymax])
    plt.legend(legend, loc='lower right')
    if filename is not None:
        plt.savefig(filename)
    plt.show()

def generate_data_for_plot(rlProblem, nEpisodes=200, nTrials=100, nSteps=100):
    '''Generate data for 100 trials of reinforce, modelBasedRL and qLearning(epsilon=0.05)
    for comparison.
    Inputs:
    rlProblem: RL2 class object
    nEpisodes: (200) number of episodes to run each algorithm
    nTrials: (100) number of trials over which we collect data
    nSteps: (100) number of steps in each episode
    Outputs:
    reinforce_avg_cumulative_reward
    modelBasedRL_avg_cumulative_reward
    qLearning_avg_cumulative_reward
    plot_legend
    '''
    #initialize variables
    reinforce_avg_cumulative_reward = np.zeros([nEpisodes])
    reinforce_avg_cumulative_reward2 = np.zeros([nEpisodes])
    modelBasedRL_avg_cumulative_reward = np.zeros([nEpisodes])
    qLearning_avg_cumulative_reward = np.zeros([nEpisodes])
    plot_legend = []

    #generate data for reinforce
    cumulative_reward = np.zeros([nTrials, nEpisodes])
    for trial in range(nTrials):
        #run reinforce for 200 episodes and 100 steps
        [Q,policy] = rlProblem.reinforce(s0=0,initialPolicyParams=np.random.rand(rlProblem.mdp.nActions,rlProblem.mdp.nStates),nEpisodes=nEpisodes,nSteps=nSteps, use_mc_est=False)
        cumulative_reward[trial,:] = rlProblem.get_reinforce_cumulative_reward()
    reinforce_avg_cumulative_reward[:] = np.mean(cumulative_reward, axis=0)
    plot_legend.append('REINFORCE (Gn)')

    #generate data for reinforce
    cumulative_reward = np.zeros([nTrials, nEpisodes])
    for trial in range(nTrials):
        #run reinforce for 200 episodes and 100 steps
        [Q,policy] = rlProblem.reinforce(s0=0,initialPolicyParams=np.random.rand(rlProblem.mdp.nActions,rlProblem.mdp.nStates),nEpisodes=nEpisodes,nSteps=nSteps, use_mc_est=True)
        cumulative_reward[trial,:] = rlProblem.get_reinforce_cumulative_reward()
    reinforce_avg_cumulative_reward2[:] = np.mean(cumulative_reward, axis=0)
    plot_legend.append('REINFORCE (V-est)')

    #generate data for modelBasedRL
    cumulative_reward = np.zeros([nTrials, nEpisodes])
    for trial in range(nTrials):
        #run modelBasedRL for 200 episodes and 100 steps
        [V,policy] = rlProblem.modelBasedRL(s0=0,defaultT=np.ones([rlProblem.mdp.nActions,rlProblem.mdp.nStates,rlProblem.mdp.nStates])/rlProblem.mdp.nStates,initialR=np.zeros([rlProblem.mdp.nActions,rlProblem.mdp.nStates]),nEpisodes=nEpisodes,nSteps=nSteps,epsilon=0.3)
        cumulative_reward[trial,:] = rlProblem.get_model_based_rl_cumulative_reward()
    modelBasedRL_avg_cumulative_reward[:] = np.mean(cumulative_reward, axis=0)
    plot_legend.append('ModelBasedRL (e=0.3)')

    #generate data for qLearning
    cumulative_reward = np.zeros([nTrials, nEpisodes])
    for trial in range(nTrials):
        #run qLearning for 200 episodes and 100 steps
        [Q,policy] = rlProblem.qLearning(s0=0,initialQ=np.zeros([rlProblem.mdp.nActions,rlProblem.mdp.nStates]),nEpisodes=nEpisodes,nSteps=nSteps,epsilon=0.05)
        cumulative_reward[trial,:] = rlProblem.get_q_learning_cumulative_reward()
    qLearning_avg_cumulative_reward[:] = np.mean(cumulative_reward, axis=0)
    plot_legend.append('qLearning (e=0.05)')

    return [reinforce_avg_cumulative_reward, reinforce_avg_cumulative_reward2, modelBasedRL_avg_cumulative_reward, qLearning_avg_cumulative_reward, plot_legend]


def generate_bandit_data_for_plot(rlProblem, nIterations=200, nTrials=1000):
    '''Generate data for 100 trials of reinforce, modelBasedRL and qLearning(epsilon=0.05)
    for comparison.
    Inputs:
    rlProblem: RL2 class object
    nIterations: (200) number of iterations to run each algorithm
    nTrials: (1000) number of trials over which we collect data
    Outputs:
    epsilon_greedy_avg_reward
    ucb_bandit_avg_reward
    thompson_sampling_avg_reward
    plot_legend
    '''
    #initialize variables
    epsilon_greedy_avg_reward = np.zeros([nIterations])
    ucb_bandit_avg_reward = np.zeros([nIterations])
    thompson_sampling_avg_reward = np.zeros([nIterations])
    plot_legend = []

    #epsilon greedy (epsilon=1/nIterations)
    cumulative_reward = np.zeros([nTrials, nIterations])
    for trial in range(nTrials):
        #run epsilon greedy for 200 iterations
        empiricalMean = rlProblem.epsilonGreedyBandit(nIterations, decay_epsilon=True)
        cumulative_reward[trial,:] = rlProblem.get_epsilon_greedy_reward()
    epsilon_greedy_avg_reward[:] = np.mean(cumulative_reward, axis=0)
    plot_legend.append('Epsilon Greedy')

    #UCB bandit
    cumulative_reward = np.zeros([nTrials, nIterations])
    for trial in range(nTrials):
        #run ucb bandit for 200 iterations
        empiricalMean = rlProblem.UCBbandit(nIterations)
        cumulative_reward[trial,:] = rlProblem.get_ucb_bandit_reward()
    ucb_bandit_avg_reward[:] = np.mean(cumulative_reward, axis=0)
    plot_legend.append('Upper Confidence Bound')

    #Thompson Sampling
    cumulative_reward = np.zeros([nTrials, nIterations])
    for trial in range(nTrials):
        #run thompson sampling for 200 iterations
        empiricalMean = rlProblem.thompsonSamplingBandit(prior = np.ones([rlProblem.mdp.nActions, 2]), nIterations=nIterations)
        cumulative_reward[trial,:] = rlProblem.get_thompson_sampling_reward()
    thompson_sampling_avg_reward[:] = np.mean(cumulative_reward, axis=0)
    plot_legend.append('Thompson Sampling')

    return [epsilon_greedy_avg_reward, ucb_bandit_avg_reward, thompson_sampling_avg_reward, plot_legend]
