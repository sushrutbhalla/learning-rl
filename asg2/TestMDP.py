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
print ("[DEBUG] valueIteration: {}".format([V,nIterations,epsilon]))
policy = mdp.extractPolicy(V)
print ("[DEBUG] extractPolicy: {}".format(policy))
[V,nIterations,epsilon] = mdp.valueIteration(initialV=np.zeros(mdp.nStates),tolerance=0.0)
print ("[DEBUG] valueIteration: {}".format([V,nIterations,epsilon]))
policy = mdp.extractPolicy(V)
print ("[DEBUG] extractPolicy: {}".format(policy))
[V,nIterations,epsilon] = mdp.valueIteration(initialV=np.zeros(mdp.nStates),nIterations=100)
print ("[DEBUG] valueIteration: {}".format([V,nIterations,epsilon]))
policy = mdp.extractPolicy(V)
print ("[DEBUG] extractPolicy: {}".format(policy))
[V,nIterations,epsilon] = mdp.valueIteration(initialV=np.zeros(mdp.nStates),nIterations=100,tolerance=0.0)
print ("[DEBUG] valueIteration: {}".format([V,nIterations,epsilon]))
policy = mdp.extractPolicy(V)
print ("[DEBUG] extractPolicy: {}".format(policy))

V = mdp.evaluatePolicy(np.array([1,0,1,0]))
print ("[DEBUG] evaluatePolicy: {}".format(V))
V = mdp.evaluatePolicy(np.array([1,0,0,0]))
print ("[DEBUG] evaluatePolicy: {}".format(V))
V = mdp.evaluatePolicy(np.array([1,1,1,0]))
print ("[DEBUG] evaluatePolicy: {}".format(V))
V = mdp.evaluatePolicy(np.array([0,1,1,1]))
print ("[DEBUG] evaluatePolicy: {}".format(V))

[policy,V,iterId] = mdp.policyIteration(np.array([0,0,0,0]))
print ("[DEBUG] policyIteration: {}".format([policy,V,iterId]))
[policy,V,iterId] = mdp.policyIteration(np.array([1,0,0,0]))
print ("[DEBUG] policyIteration: {}".format([policy,V,iterId]))
[policy,V,iterId] = mdp.policyIteration(np.array([1,1,1,0]))
print ("[DEBUG] policyIteration: {}".format([policy,V,iterId]))

[V,iterId,epsilon] = mdp.evaluatePolicyPartially(np.array([1,0,1,0]),np.array([0,10,0,13]))
print ("[DEBUG] evaluatePolicyPartially: {}".format([V,iterId,epsilon]))
[V,iterId,epsilon] = mdp.evaluatePolicyPartially(np.array([1,1,1,0]),np.array([10,0,15,1]))
print ("[DEBUG] evaluatePolicyPartially: {}".format([V,iterId,epsilon]))

[policy,V,iterId,tolerance] = mdp.modifiedPolicyIteration(np.array([1,0,1,0]),np.array([0,10,0,13]))
print ("[DEBUG] modifiedPolicyIteration: {}".format([policy,V,iterId,tolerance]))
[policy,V,iterId,tolerance] = mdp.modifiedPolicyIteration(np.array([1,0,1,0]),np.array([0,10,0,13]),nEvalIterations=np.inf) #policy iteration
print ("[DEBUG] modifiedPolicyIteration: {}".format([policy,V,iterId,tolerance]))
[policy,V,iterId,tolerance] = mdp.modifiedPolicyIteration(np.zeros(mdp.nStates,dtype=int),np.zeros(mdp.nStates),nEvalIterations=np.inf, tolerance=0.0)
print ("[DEBUG] modifiedPolicyIteration: {}".format([policy,V,iterId,tolerance]))
[policy,V,iterId,tolerance] = mdp.modifiedPolicyIteration(np.array([1,0,1,0]),np.array([0,10,0,13]),nEvalIterations=1)
print ("[DEBUG] modifiedPolicyIteration: {}".format([policy,V,iterId,tolerance]))
[policy,V,iterId,tolerance] = mdp.modifiedPolicyIteration(np.zeros(mdp.nStates,dtype=int),np.zeros(mdp.nStates),nEvalIterations=0, tolerance=0.0) #value iteration
print ("[DEBUG] modifiedPolicyIteration: {}".format([policy,V,iterId,tolerance]))
[policy,V,iterId,tolerance] = mdp.modifiedPolicyIteration(np.zeros(mdp.nStates,dtype=int),np.zeros(mdp.nStates),nEvalIterations=0) #value iteration
print ("[DEBUG] modifiedPolicyIteration: {}".format([policy,V,iterId,tolerance]))