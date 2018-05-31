from __future__ import division
import random
import copy
import math
import decimal
import numpy as np
import MDP

class RL:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward
        self.debug = False

    def sampleRewardAndNextState(self,state,action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs:
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action,state])
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]

    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0):
        '''qLearning algorithm.  Epsilon exploration and Boltzmann exploration
        are combined in one procedure by sampling a random action with
        probabilty epsilon and performing Boltzmann exploration otherwise.
        When epsilon and temperature are set to 0, there is no exploration.

        Inputs:
        s0 -- initial state
        initialQ -- initial Q function (|A|x|S| array)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random
        temperature -- parameter that regulates Boltzmann exploration

        Outputs:
        Q -- final Q function (|A|x|S| array)
        policy -- final policy
        '''

        # # temporary values to ensure that the code compiles until this
        # # function is coded
        # Q = np.zeros([self.mdp.nActions,self.mdp.nStates])
        # policy = np.zeros(self.mdp.nStates,int)

        assert initialQ.ndim == 2, "Invalid initialV: it has dimensionality " + repr(initialQ.ndim)
        assert initialQ.shape[0] == self.mdp.nActions and initialQ.shape[1] == self.mdp.nStates, \
                "Invalid initialV: it has shape " + repr(initialQ.shape)
        assert s0 < self.mdp.nStates, "Invalid s0: value is " + repr(s0)

        Q = initialQ
        state = s0
        state_p = s0
        action = random.randint(0, self.mdp.nActions-1) #random initialization of valid action
        reward = self.mdp.R[action, state]
        N = np.zeros([self.mdp.nActions, self.mdp.nStates])
        prob_act = np.zeros([self.mdp.nActions])
        lr = 1
        #changeInQ = True
        iterId = 0
        policy = np.zeros(self.mdp.nStates,int)
        #while changeInQ:
        for _ in range(nEpisodes):
            action_chosen = False
            #select an action based on epsilon (exploration vs exploitation)
            if epsilon != 0. and random.random() <= epsilon:
                #explore with probability epsilon
                action = random.randint(0, self.mdp.nActions-1)
                action_chosen = True
                if self.debug:
                    print ("[exploration] action: {}".format(action))
                    print ("self.mdp.nActions: {}".format(self.mdp.nActions))
            if (not action_chosen) and temperature != 0:
                #select an action based on boltzmann exploration
                denominator = 0
                for act_idx in range(self.mdp.nActions):
                    denominator += math.exp(Q[act_idx, state]/temperature)
                for act_idx in range(self.mdp.nActions):
                    prob_act[act_idx] = math.exp(Q[act_idx, state]/temperature)/denominator
                assert round(decimal.Decimal(np.sum(prob_act)),2) == 1., "total probabilities don't equal 1: " + repr(np.sum(prob_act))
                action = np.random.choice(self.mdp.nActions, p=prob_act)
                action_chosen = True
            if not action_chosen:
                #exploit the best action from the policy
                action = np.argmax(Q[:, state], axis=0)
                assert np.argmax(Q[:, state], axis=0) == np.argmax(Q, axis=0)[state], \
                        "Wrong action found during exploitation: " + repr(action)
                action_chosen = True
                if self.debug:
                    print ("[exploitation] action: {}".format(action))
                    print ("Q: {}".format(Q))
                    print ("Q.shape {}".format(Q.shape))
                    print ("Q[0, :] {}".format(Q[0, :]))
                    print ("Q[:, 0] {}".format(Q[:, 0]))
                    print ("np.argmax(Q[:, state], axis=0): {}".format(np.argmax(Q[:, state], axis=0)))
                    print ("np.argmax(Q, axis=0)[state]: {}".format(np.argmax(Q, axis=0)[state]))
            #observe state_p and reward
            reward, state_p = self.sampleRewardAndNextState(state, action)
            #update counter for (state, action) in N
            N[action, state] += 1
            #learning rate
            lr = 1./N[action, state]
            #update Q-value
            Q_new = copy.deepcopy(Q)
            #action_p is the best action from state_p under this Q function
            action_p = np.argmax(Q[:, state_p], axis=0)
            Q_new[action, state]= Q[action, state] + lr*(reward+(self.mdp.discount*Q[action_p, state_p])-Q[action, state])
            if np.array_equal(Q, Q_new):
                #changeInQ = False
                #break from the loop as the value has converged
                break
            state = state_p
            Q = Q_new
            iterId += 1
            if iterId == nSteps:
                iterId = 0
                state = s0
                if self.debug:
                    print ("\n===============================================================")
                    print ("N: {}".format(N))
                    print ("Q: {}".format(Q))
                    print ("policy: {}".format(np.argmax(Q, axis=0)))
                    print ("lr: {}".format(lr))
                    print ("update: {}".format(Q_new - Q))
        #TODO does the policy improvement step need to go in the for loop?
        #but from page 80 of Sutton book, the use of max q_k to evaluate q_(k+1) is same as using argmax policy_k
        policy = np.argmax(Q, axis=0)

        return [Q,policy]