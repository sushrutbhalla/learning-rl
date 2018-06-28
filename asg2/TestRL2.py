import numpy as np
import MDP
import RL2


def sampleBernoulli(mean):
    ''' function to obtain a sample from a Bernoulli distribution

    Input:
    mean -- mean of the Bernoulli

    Output:
    sample -- sample (0 or 1)
    '''

    if np.random.rand(1) < mean: return 1
    else: return 0


# Multi-arm bandit problems (3 arms with probabilities 0.3, 0.5 and 0.7)
T = np.array([[[1]],[[1]],[[1]]])
R = np.array([[0.3],[0.5],[0.7]])
discount = 0.999
mdp = MDP.MDP(T,R,discount)
banditProblem = RL2.RL2(mdp,sampleBernoulli)

# Test epsilon greedy strategy
empiricalMeans = banditProblem.epsilonGreedyBandit(nIterations=200)
print ("\nepsilonGreedyBandit results")
print (empiricalMeans)

# Test Thompson sampling strategy
empiricalMeans = banditProblem.thompsonSamplingBandit(prior=np.ones([mdp.nActions,2]),nIterations=200)
print ("\nthompsonSamplingBandit results")
print (empiricalMeans)

# Test UCB strategy
empiricalMeans = banditProblem.UCBbandit(nIterations=200)
print ("\nUCBbandit results")
print (empiricalMeans)

#TODO before submission compare the files with the files on server to make sure that the function parameters are correct
''' Construct simple MDP as described in Lecture 2a Slides 13-14'''
T = np.array([[[0.5,0.5,0,0],[0,1,0,0],[0.5,0.5,0,0],[0,1,0,0]],[[1,0,0,0],[0.5,0,0,0.5],[0.5,0,0.5,0],[0,0,0.5,0.5]]])
R = np.array([[0,0,10,10],[0,0,10,10]])
discount = 0.9
mdp = MDP.MDP(T,R,discount)
rlProblem = RL2.RL2(mdp,np.random.normal)
#I think advertise is 0 and save is 1

# Test REINFORCE
[Q,policy] = rlProblem.reinforce(s0=0,initialPolicyParams=np.random.rand(mdp.nActions,mdp.nStates),nEpisodes=1000,nSteps=100)
print ("\nREINFORCE results")
print (Q)
print (policy)
print ("rounded policy: \n{}".format(np.round(policy, decimals=1)))
print ("last 10 episode rewards: {}".format(rlProblem.get_reinforce_cumulative_reward()[-10:]))
#mean episode reward doesn't mean much
# print ("mean cumulative reward: {}".format(np.mean(rlProblem.get_reinforce_cumulative_reward())))

# Test model-based RL
[V,policy] = rlProblem.modelBasedRL(s0=0,defaultT=np.ones([mdp.nActions,mdp.nStates,mdp.nStates])/mdp.nStates,initialR=np.zeros([mdp.nActions,mdp.nStates]),nEpisodes=100,nSteps=100,epsilon=0.3)
print ("\nmodel-based RL results")
print (V)
print (policy)

# for i in range(200):
# 	[Q,policy] = rlProblem.reinforce(s0=0,initialPolicyParams=np.random.rand(mdp.nActions,mdp.nStates),nEpisodes=1000,nSteps=100)
# 	print ("mean cumulative reward: {}".format(np.mean(rlProblem.get_reinforce_cumulative_reward())))