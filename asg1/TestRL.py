import numpy as np
import MDP
import RL


''' Construct simple MDP as described in Lecture 2a Slides 13-14'''
T = np.array([[[0.5,0.5,0,0],[0,1,0,0],[0.5,0.5,0,0],[0,1,0,0]],[[1,0,0,0],[0.5,0,0,0.5],[0.5,0,0.5,0],[0,0,0.5,0.5]]])
R = np.array([[0,0,10,10],[0,0,10,10]])
discount = 0.9
mdp = MDP.MDP(T,R,discount)
rlProblem = RL.RL(mdp,np.random.normal)

# Test Q-learning
[Q,policy] = rlProblem.qLearning(s0=0,initialQ=np.zeros([mdp.nActions,mdp.nStates]),nEpisodes=1000,nSteps=100,epsilon=0.3)
print "\nQ-learning results"
print Q
print policy
#TODO check with lecture video if this could converge or ask Professor or what is the meaning of episode and nsteps? as the q-learning doesn't have a steps, it has one-sample approximation
#TODO check if there are any other properties we can extract from this rl problem
#TODO what is the convergence for Q-learning because it wasn't mentioned
#TODO the capital Q mentioned in the slides is meant for per state,action and not a matrix operation, right?
[Q,policy] = rlProblem.qLearning(s0=0,initialQ=np.zeros([mdp.nActions,mdp.nStates]),nEpisodes=1000,nSteps=100,epsilon=0., temperature=100)
print "\nQ-learning results"
print Q
print policy
