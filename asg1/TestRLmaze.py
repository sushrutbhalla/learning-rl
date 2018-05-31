import numpy as np
import MDP
import RL

import sys

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
  and 4 actions (up, down, right, left).'''

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
rlProblem = RL.RL(mdp,np.random.normal)

def check_for_goodness(policy):
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
  #print ("Bad States: {}".format(bad_states))
  return False, bad_states

def turn_towards_pit(policy):
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

def policy_reach_terminal(policy, s0=0, nSteps=100):
  state = s0
  total_reward = 0
  # print ("policy shape: {}\npolicy: {}".format(policy.shape, policy))
  for idx in range(nSteps):
    action = policy[state]
    reward, state_p = rlProblem.sampleRewardAndNextState(state, action)
    # print ("[DEBUG] state, action: {}, {}".format(state, action))
    # print ("[DEBUG] reward, state: {}, {}".format(reward, state_p))
    state = state_p
    total_reward += reward
    if state == 16:
      return True, total_reward
  return False, total_reward

def print_policy_word(policy, epsilon):
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
  terminate, total_reward = policy_reach_terminal(policy)
  print ("\n------------------------------ epsilon: {} -------------------------------------".format(epsilon))
  print ("TERMINATE: {}, GOODNESS: {}, SOFT_GOODNESS: {}\nbad states: {} Total Reward: {}\n{}".format(\
    terminate, goodness, not turn_towards_pit(policy), bad_states,\
    total_reward, policy_word))
  return

# Test Q-learning
[Q,policy] = rlProblem.qLearning(s0=0,initialQ=np.zeros([mdp.nActions,mdp.nStates]),nEpisodes=200,nSteps=100,epsilon=0.05)
print_policy_word(policy, epsilon=0.05)
#print ("\nQ-learning results")
#print (Q)
#print (policy)
[Q,policy] = rlProblem.qLearning(s0=0,initialQ=np.zeros([mdp.nActions,mdp.nStates]),nEpisodes=200,nSteps=100,epsilon=0.1)
print_policy_word(policy, epsilon=0.1)
#print ("\nQ-learning results")
#print (Q)
# print (policy)
[Q,policy] = rlProblem.qLearning(s0=0,initialQ=np.zeros([mdp.nActions,mdp.nStates]),nEpisodes=200,nSteps=100,epsilon=0.3)
print_policy_word(policy, epsilon=0.3)
#print ("\nQ-learning results")
#print (Q)
# print (policy)
[Q,policy] = rlProblem.qLearning(s0=0,initialQ=np.zeros([mdp.nActions,mdp.nStates]),nEpisodes=200,nSteps=100,epsilon=0.5)
print_policy_word(policy, epsilon=0.5)
#print ("\nQ-learning results")
#print (Q)
# print (policy)


#DEBUG TODO remove
for idx in range(100):
  for esps in ([0.05, 0.1, 0.3, 0.5]):
    [Q,policy] = rlProblem.qLearning(s0=0,initialQ=np.zeros([mdp.nActions,mdp.nStates]),nEpisodes=10000,nSteps=100,epsilon=esps,temperature=0)
    if not policy_reach_terminal(policy):
      print ("Policy didn't terminate: ({},{}) {}".format(idx, esps, policy))
      break
  else:
    continue
  break

[Q,policy] = rlProblem.qLearning(s0=0,initialQ=np.zeros([mdp.nActions,mdp.nStates]),nEpisodes=10000,nSteps=100,epsilon=0.3,temperature=0)
print_policy_word(policy, epsilon=0.3)
