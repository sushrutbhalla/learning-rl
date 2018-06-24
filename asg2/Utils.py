import numpy as np

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