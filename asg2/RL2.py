import numpy as np
import MDP
import math

class RL2:
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
        self.debug = True

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

    def sampleSoftmaxPolicy(self,policyParams,state):
        '''Procedure to sample an action from stochastic policy
        pi(a|s) = exp(policyParams(a,s))/[sum_a' exp(policyParams(a',s))])
        This function should be called by reinforce() to selection actions

        Inputs:
        policyParams -- parameters of a softmax policy (|A|x|S| array)
        state -- current state

        Outputs:
        action -- sampled action
        '''

        # temporary value to ensure that the code compiles until this
        # function is coded
        # action = 0

        #shift the policyParams to avoid NaN
        shifted_policyParams = policyParams[:, state] - np.max(policyParams[:, state])
        act_dist = np.exp(shifted_policyParams)/np.sum(np.exp(shifted_policyParams))
        assert len(act_dist) == self.mdp.nActions, "Length of stochastic actions for state don't match"
        #keeping the assert sensitive to the way python handles decimal values
        assert np.sum(act_dist) >= 0.9999 and np.sum(act_dist) <= 1.0001, "Softmax sum doesn't equal 1: " + repr(np.sum(act_dist))
        action = np.random.choice(self.mdp.nActions, p=act_dist)

        return action

    def modelBasedRL(self,s0,defaultT,initialR,nEpisodes,nSteps,epsilon=0):
        '''Model-based Reinforcement Learning with epsilon greedy
        exploration.  This function should use value iteration,
        policy iteration or modified policy iteration to update the policy at each step

        Inputs:
        s0 -- initial state
        defaultT -- default transition function when a state-action pair has not been vsited
        initialR -- initial estimate of the reward function
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random

        Outputs:
        V -- final value function
        policy -- final policy
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        V = np.zeros(self.mdp.nStates)
        policy = np.zeros(self.mdp.nStates,int)

        return [V,policy]

    def epsilonGreedyBandit(self,nIterations):
        '''Epsilon greedy algorithm for bandits (assume no discount factor)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs:
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nStates)

        return empiricalMeans

    def thompsonSamplingBandit(self,prior,nIterations,k=1):
        '''Thompson sampling algorithm for Bernoulli bandits (assume no discount factor)

        Inputs:
        prior -- initial beta distribution over the average reward of each arm (|A|x2 matrix such that prior[a,0] is the alpha hyperparameter for arm a and prior[a,1] is the beta hyperparameter for arm a)
        nIterations -- # of arms that are pulled
        k -- # of sampled average rewards

        Outputs:
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nStates)

        return empiricalMeans

    def UCBbandit(self,nIterations):
        '''Upper confidence bound algorithm for bandits (assume no discount factor)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs:
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nStates)

        return empiricalMeans

    def reinforce(self,s0,initialPolicyParams,nEpisodes,nSteps):
        '''reinforce algorithm.  Learn a stochastic policy of the form
        pi(a|s) = exp(policyParams(a,s))/[sum_a' exp(policyParams(a',s))]).
        This function should call the function sampleSoftmaxPolicy(policyParams,state) to select actions

        Inputs:
        s0 -- initial state
        initialPolicyParams -- parameters of the initial policy (array of |A|x|S| entries)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0)
        nSteps -- # of steps per episode

        Outputs:
        policyParams -- parameters of the final policy (array of |A|x|S| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        policyParams = np.zeros((self.mdp.nActions,self.mdp.nStates))

        #setup variables
        policyParams = initialPolicyParams
        lr = 1
        N = np.zeros([self.mdp.nActions, self.mdp.nStates])
        #TODO remove comment: what is the iniital value of the Gn? can it converge from any value?
        Gn = np.zeros([self.mdp.nActions, self.mdp.nStates])
        sch_policy = np.zeros([self.mdp.nActions, self.mdp.nStates])
        policy = np.zeros([self.mdp.nActions, self.mdp.nStates])
        #format for episode_path: (state, action, reward)
        episode_path = np.zeros([nSteps, 3])
        state = s0
        for episode_idx in range(nEpisodes):
            #loop through the entire episode and generate trajectory
            for step_idx in range(nSteps):
                action = self.sampleSoftmaxPolicy(policyParams, state)
                reward, state_p = self.sampleRewardAndNextState(state, action)
                episode_path[step_idx,:] = [state,action,reward]
                state = state_p
            #loop through the entire episode and evaluate the policy and update policy parameters
            for step_idx in range(nSteps):
                state = int(episode_path[step_idx, 0])
                action = int(episode_path[step_idx, 1])
                N[action, state] += 1
                lr = 1./N[action, state]
                Gn[action, state] = 0.
                for idx in range(nSteps-step_idx):
                    reward = episode_path[step_idx+idx, 2]
                    Gn[action, state] += (self.mdp.discount**idx)*reward
                #update the policy parameters using the update equation
                #\theta <- \theta + \alpha*\gamma^n*Gn*\nabla(log \pi_\theta (a_n|s_n) )
                #the derivative of log of softmax function is:
                #   1-softmax(i) when i=j (diagonal entries in jacobian)
                #   -softmax(j) when i!=j (all other entries in jacobian)
                #so we don't need to compute the log and derivative, we only need to compute the softmax for each index
                # get the policy parameters for this state as that is all that will get updated
                pp_state = policyParams[:, state] #TODO change this to shifted policy params and see change in output
                #                                  shifted_policyParams = policyParams[:, state] - np.max(policyParams[:, state])
                value_state = Gn[:, state]
                pi_state = sch_policy[:, state] #TODO I don't think this is ever needed
                #initialize jacobian to |A|x|A| as the policy parameter is of size |A| and so is the policy \pi
                jacobian = np.zeros([self.mdp.nActions, self.mdp.nActions])
                for col in range(jacobian.shape[1]):
                    for row in range(jacobian.shape[0]):
                        if col == row:
                            jacobian[row][col] = 1 - math.exp(pp_state[col])/np.sum(np.exp(pp_state[:]))
                        else:
                            jacobian[row][col] = 0 - math.exp(pp_state[col])/np.sum(np.exp(pp_state[:]))
                #update the policy parameters
                policyParams[:, state] = pp_state + lr*(self.mdp.discount**step_idx)*np.matmul(jacobian, np.transpose(value_state))
            if self.debug and episode_idx % 100 == 0:
                print ("episode_idx: {}".format(episode_idx))
                print ("policyParams: \n{}".format(policyParams))
                print ("Gn: \n{}".format(Gn))

        #compute the policy from the policy parameters
        for state in range(self.mdp.nStates):
            # TODO check if it works first and then: shifted_policyParams = policyParams[:, state] - np.max(policyParams[:, state])
            pp_state = policyParams[:, state]
            policy[:, state] = np.exp(pp_state)/np.sum(np.exp(pp_state))

        return policyParams, policy
