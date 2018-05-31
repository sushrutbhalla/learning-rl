from MDP import *

''' Construct simple MDP as described in Lecture 2a Slides 13-14'''
# Transition function: |A| x |S| x |S'| array
T = np.array([[[0.5,0.5,0,0],[0,1,0,0],[0.5,0.5,0,0],[0,1,0,0]],[[1,0,0,0],[0.5,0,0,0.5],[0.5,0,0.5,0],[0,0,0.5,0.5]]])
# Reward function: |A| x |S| array
R = np.array([[0,0,10,10],[0,0,10,10]])
# Discount factor: scalar in [0,1)
discount = 0.9
# MDP object
mdp = MDP(T,R,discount)

'''Test each procedure'''
[V,nIterations,epsilon] = mdp.valueIteration(initialV=np.zeros(mdp.nStates))
#TODO do code check with different values of tolerance and nIteration for valueIteration and policyIteration
print ("[DEBUG] valueIteration: {}".format([V,nIterations,epsilon]))
policy = mdp.extractPolicy(V)
print ("[DEBUG] extractPolicy: {}".format(policy))
V = mdp.evaluatePolicy(np.array([1,0,1,0]))
print ("[DEBUG] evaluatePolicy: {}".format(V))
[policy,V,iterId] = mdp.policyIteration(np.array([0,0,0,0]))
print ("[DEBUG] policyIteration: {}".format([policy,V,iterId]))
[V,iterId,epsilon] = mdp.evaluatePolicyPartially(np.array([1,0,1,0]),np.array([0,10,0,13]))
#TODO try different inputs
print ("[DEBUG] evaluatePolicyPartially: {}".format([V,iterId,epsilon]))
[policy,V,iterId,tolerance] = mdp.modifiedPolicyIteration(np.array([1,0,1,0]),np.array([0,10,0,13]))
#TODO check modified policy iteration with k=0 and k=inf if we get value iteration and policy iteration results
print ("[DEBUG] modifiedPolicyIteration: {}".format([policy,V,iterId,tolerance]))
