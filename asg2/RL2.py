from __future__ import division, print_function
import numpy as np
import MDP
import random
import copy
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
        pp_state = policyParams[:, state] # - np.max(policyParams[:, state])
        act_dist = np.exp(pp_state)/np.sum(np.exp(pp_state))
        assert len(act_dist) == self.mdp.nActions, "Length of stochastic actions for state don't match"
        #keeping the assert sensitive to the way python handles decimal values
        # assert np.sum(act_dist) >= 0.9999 and np.sum(act_dist) <= 1.0001, "Softmax sum doesn't equal 1: " + repr(np.sum(act_dist))
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
        # V = np.zeros(self.mdp.nStates)
        # policy = np.zeros(self.mdp.nStates,int)

        #initialize variables for ModelBasedRL
        model_T = defaultT  #TODO confirm shape: |A|x|S|x|S|
        model_R = initialR  #TODO confirm shape: |A|x|S|
        V = np.zeros([self.mdp.nStates])
        #the counters should start at 1 because the transition model already has a default
        #probability of 1/|S| TODO
        #I don't think so. but what is the point of defaultT and defaultR?
        N_doubles = np.zeros([self.mdp.nActions, self.mdp.nStates])
        N_triples = np.zeros([self.mdp.nActions, self.mdp.nStates, self.mdp.nStates])
        policy = np.zeros([self.mdp.nStates], dtype=int)
        for episode_idx in range(nEpisodes):
            state = s0 
            for step_idx in range(nSteps):
                #select an action based on epsilon (exploration vs exploitation)
                if epsilon > 0. and random.random() <= epsilon:
                    #explore with probability epsilon
                    action = random.randint(0, self.mdp.nActions-1)
                    if self.debug:
                        print ("[exploration] action: {}".format(action))
                        print ("self.mdp.nActions: {}".format(self.mdp.nActions))
                else:
                    #generate policy using extractPolicy TODO check if this is OK
                    policy = self.mdp.extractPolicy(V,use_default=False,R_mdp=model_R,T_mdp=model_T)
                    action = policy[state]
                    if self.debug:
                        print ("[exploitation] action: {}".format(action))
                        print ("self.mdp.nActions: {}".format(self.mdp.nActions))
                #generate next state, reward
                reward, state_p = self.sampleRewardAndNextState(state, action)

                #update counts
                N_doubles[action, state] += 1
                N_triples[action, state, state_p] += 1
                #update transitions
                for state_p_idx in range(self.mdp.nStates):
                    model_T[action, state, state_p_idx] = N_triples[action, state, state_p_idx]*1./N_doubles[action, state]
                #update reward
                model_R[action, state] = (reward + (N_doubles[action, state]-1)*model_R[action, state])/N_doubles[action, state]
                #solve value iteration using current model_R and model_T
                V,_,_ = self.mdp.valueIteration(V, tolerance=0.01,use_default=False,R_mdp=model_R,T_mdp=model_T)

                #update the state
                state = state_p

                #TODO add convergence step
            if self.debug and episode_idx % 100 == 0:
                print ("episode: {}".format(episode_idx))
                print ("N_doubles: \n{}".format(N_doubles))
                print ("N_triples: \n{}".format(N_triples))
                print ("model_T: \n{}".format(model_T))
                print ("model_R: \n{}".format(model_R))

        if self.debug:
            print ("N_doubles: \n{}".format(N_doubles))
            print ("N_triples: \n{}".format(N_triples))
            print ("model_T: \n{}".format(model_T))
            print ("model_R: \n{}".format(model_R))

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
        # empiricalMeans = np.zeros(self.mdp.nStates)

        #TODO starting with random value of epsilon
        epsilon = 0.3
        #TODO check if this hold all the time
        assert self.mdp.nStates == 1, "We can only have 1 state in bandits problem. Given: " + repr(self.mdp.nStates)
        #initialize variables
        state = 0    #state is fixed at 0
        empiricalMeans = np.zeros([self.mdp.nActions])
        N = np.zeros([self.mdp.nActions])
        for iteration in range(nIterations):
            if epsilon > 0. and random.random() <= epsilon:
                action = random.randint(0, self.mdp.nActions-1)
                # if self.debug:
                #     print ("[exploration] action: {}".format(action))
                #     print ("self.mdp.nActions: {}".format(self.mdp.nActions))
            else:
                action = np.argmax(empiricalMeans)
                # if self.debug:
                #     print ("[exploitation] action: {}".format(action))
                #     print ("self.mdp.nActions: {}".format(self.mdp.nActions))
            #sample reward for this action
            reward,_ = self.sampleRewardAndNextState(state, action) #TODO check is this correct?
            N[action] += 1
            empiricalMeans[action] = empiricalMeans[action] + (1./N[action])*(reward - empiricalMeans[action])

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
        # empiricalMeans = np.zeros(self.mdp.nStates)

        #TODO check if this hold all the time
        assert self.mdp.nStates == 1, "We can only have 1 state in bandits problem. Given: " + repr(self.mdp.nStates)
        #initialize variables
        state = 0    #state is fixed at 0
        alpha = np.ones([self.mdp.nActions])
        beta = np.ones([self.mdp.nActions])
        samples = np.zeros([self.mdp.nActions, k])
        mean_reward = np.zeros([self.mdp.nActions])
        V_reward = np.zeros([self.mdp.nActions])
        Na = np.zeros([self.mdp.nActions])
        empiricalMeans = np.zeros([self.mdp.nActions])
        for iteration in range(nIterations):
            #generate samples for all actions from prior beta distribution
            for action_idx in range(self.mdp.nActions):
                samples[action_idx, :] = np.random.beta(alpha[action_idx], beta[action_idx], k)
                mean_reward[action_idx] = np.mean(samples[action_idx,:])
            action = np.argmax(mean_reward)
            reward,_ = self.sampleRewardAndNextState(state, action) #TODO check is this correct?
            # if self.debug:
            #     print ("action: {}, reward: {}".format(action, reward))
            #     print ("hoeffding_term: {}".format(hoeffding_term))
            V_reward[action] += reward 
            Na[action] += 1
            #update posterior beta distribution 
            if reward == 1: alpha[action] += 1
            else: beta[action] += 1

        empiricalMeans = np.divide(V_reward, Na)
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
        # empiricalMeans = np.zeros(self.mdp.nStates)

        #TODO check if this hold all the time
        assert self.mdp.nStates == 1, "We can only have 1 state in bandits problem. Given: " + repr(self.mdp.nStates)
        #initialize variables
        state = 0    #state is fixed at 0
        #TODO start the value of emperical means at a high value so that all actions get a chance to be selected
        #keep it at one because the Na is also initialized to 1, this way we can say that the
        #initial reward was 1 and not 0
        #TODO or should we just change this and run all actions once and start off the initial estimate
        empiricalMeans = np.ones([self.mdp.nActions])
        hoeffding_term = np.ones([self.mdp.nActions])
        #TODO starting the count at 1 will still give the same results as the argmax
        #will choose the action based on which one is updated
        Na = np.ones([self.mdp.nActions])
        N = self.mdp.nActions
        for iteration in range(nIterations):
            for action_idx in range(self.mdp.nActions):
                hoeffding_term[action_idx] = empiricalMeans[action_idx] + math.sqrt(2*math.log(N)/Na[action_idx])
            action = np.argmax(hoeffding_term)
            reward,_ = self.sampleRewardAndNextState(state, action) #TODO check is this correct?
            # if self.debug:
            #     print ("action: {}, reward: {}".format(action, reward))
            #     print ("hoeffding_term: {}".format(hoeffding_term))
            Na[action] += 1
            N += 1
            empiricalMeans[action] = empiricalMeans[action] + (1./Na[action])*(reward - empiricalMeans[action])

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
        # policyParams = initialPolicyParams
        lr = 1
        N = np.zeros([self.mdp.nActions, self.mdp.nStates])
        policy = np.zeros([self.mdp.nActions, self.mdp.nStates])
        #execute all episodes
        for episode_idx in range(nEpisodes):
            #initialize start state
            state = s0
            #format for episode_path: (state, action, reward)
            episode_path = np.zeros([nSteps, 3])
            #loop through the entire episode and generate trajectory
            for step_idx in range(nSteps):
                action = self.sampleSoftmaxPolicy(policyParams, state)
                reward, state_p = self.sampleRewardAndNextState(state, action)
                episode_path[step_idx,:] = np.array([state,action,reward])
                state = state_p
            
            #loop through the entire episode and evaluate the policy and update policy parameters
            for step_idx in range(nSteps):
                state = int(episode_path[step_idx, 0])
                action = int(episode_path[step_idx, 1])
                N[action, state] += 1
                lr = 1./N[action, state]
                #compute Gn estimate for update
                Gn_upd = 0.
                for idx in range(step_idx, nSteps):
                    reward = episode_path[idx, 2]
                    Gn_upd += (self.mdp.discount**(idx-step_idx))*reward

                #update the policy parameters using the update equation
                #\theta <- \theta + \alpha*\gamma^n*Gn*\nabla(log \pi_\theta (a_n|s_n) )
                #the derivative of log of softmax function is:
                #   1-softmax(i) when i=j (diagonal entries in jacobian)
                #   -softmax(j) when i!=j (all other entries in jacobian)
                #so we don't need to compute the log and derivative, we only need to compute the softmax for each index
                # get the policy parameters for this state as that is all that will get updated
                pp_state = policyParams[:, state] #TODO change this to shifted policy params and see change in output
                #initialize jacobian to |A|
                jacobian = np.zeros([self.mdp.nActions])
                for idx in range(jacobian.shape[0]):
                    if idx == action:
                        jacobian[idx] = 1 - np.exp(pp_state[idx])/np.sum(np.exp(pp_state))
                    else:
                        jacobian[idx] = 0 - np.exp(pp_state[idx])/np.sum(np.exp(pp_state))
                #update the policy parameters
                update_term = np.multiply(jacobian, lr*(self.mdp.discount**step_idx)*Gn_upd)
                policyParams[:, state] = policyParams[:, state] + update_term

            if self.debug and episode_idx % 100 == 0:
                print ("episode_idx: {}".format(episode_idx))
                # print ("episode_path: \n{}".format(episode_path))
                print ("policyParams: \n{}".format(policyParams))
                print ("Gn_upd: \n{}".format(Gn_upd))
                print ("jacobian {}".format(jacobian))
                print ("update_term for action,state({},{}): {}".format(action, state, update_term))

        #compute the policy from the policy parameters
        for state in range(self.mdp.nStates):
            # TODO check if it works first and then: shifted_policyParams = policyParams[:, state] - np.max(policyParams[:, state])
            pp_state = policyParams[:, state]
            policy[:, state] = np.exp(pp_state)/np.sum(np.exp(pp_state))

        return policyParams, policy


# V_est = np.zeros([self.mdp.nActions, self.mdp.nStates])
# N_V_est = np.zeros([self.mdp.nActions, self.mdp.nStates]) 
# 
# #based on the episode trajectory generated, update the value estimates of the states visited
# if use_mc_est:
#     if first_visit_mce:
#         visited_states = np.unique(episode_path[:,0]).astype(int)
#         for visit_state in visited_states:
#             state_idx = episode_path[:,0].tolist().index(visit_state)
#             visit_action = int(episode_path[state_idx, 1])
#             Gn_scalar = 0.
#             for idx in range(state_idx, nSteps):
#                 Gn_scalar += episode_path[idx, 2]*(self.mdp.discount**(idx-state_idx))
#             assert Gn_scalar != 0., "Gn_scalar is 0. even after update"
#             N_V_est[visit_action, visit_state] += 1
#             #compute new average based on Monte Carlo estimate from lecture 3b slide 11
#             V_est[visit_action, visit_state] = V_est[visit_action, visit_state] + \
#                 (1./N_V_est[visit_action, visit_state])*(Gn_scalar-V_est[visit_action, visit_state])
#             assert V_est[visit_action, visit_state] != 0., "V_est can't be 0. after update " + repr(Gn_scalar)
#     else:
#         visited_states = np.unique(episode_path[:,0]).astype(int)
#         debug_total_updates = 0
#         for visit_state in visited_states:
#             # state_idx = episode_path[:,0].tolist().index(visit_state)
#             state_indicies = np.where(episode_path[:,0].astype(int) == visit_state)[0]
#             for state_idx in state_indicies:
#                 visit_action = int(episode_path[state_idx, 1])
#                 Gn_scalar = 0.
#                 for idx in range(state_idx, nSteps):
#                     Gn_scalar += episode_path[idx, 2]*(self.mdp.discount**(idx-state_idx))
#                 assert Gn_scalar != 0., "Gn_scalar is 0. even after update"
#                 N_V_est[visit_action, visit_state] += 1
#                 #compute new average based on Monte Carlo estimate from lecture 3b slide 11
#                 V_est[visit_action, visit_state] = V_est[visit_action, visit_state] + \
#                     (1./N_V_est[visit_action, visit_state])*(Gn_scalar-V_est[visit_action, visit_state])
#                 assert V_est[visit_action, visit_state] != 0., "V_est can't be 0. after update " + repr(Gn_scalar)
#                 debug_total_updates += 1
#         assert debug_total_updates == nSteps, "total number of updates don't equal nSteps " + repr(debug_total_updates)