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
        self.reinforce_cumulative_reward = []
        self.model_based_rl_cumulative_reward = []
        self.q_learning_cumulative_reward = []
        self.epsilon_greedy_reward = []
        self.ucb_bandit_reward = []
        self.thompson_sampling_reward = []
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
        pp_state = shifted_policyParams
        act_dist = np.exp(pp_state)/np.sum(np.exp(pp_state))
        # print ("\npp_state-------------------------- {}".format(pp_state)) #TODO remove
        # print ("act_dist-------------------------- {}".format(act_dist))
        # print ("exp value:------------------------ {}".format(np.exp(pp_state)))
        assert np.count_nonzero(np.isnan(act_dist)) == 0, "NaN present in action probability for REINFORCE \naction_distribution: " +\
            repr(act_dist) + "\nPolicyParams: " + repr(pp_state) + "\nExp(PolicyParams): " + repr(np.exp(pp_state))
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
        self.model_based_rl_cumulative_reward = []
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
            cumulative_reward_episode = 0.
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
                if self.debug:
                    print ("reward: {}".format(reward))
                    print ("cumulative_reward: {}".format(reward*(self.mdp.discount**episode_step)))
                cumulative_reward_episode += reward*(self.mdp.discount**step_idx)

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

            #end of episode
            if self.debug and episode_idx % 100 == 0:
                print ("episode: {}".format(episode_idx))
                print ("N_doubles: \n{}".format(N_doubles))
                print ("N_triples: \n{}".format(N_triples))
                print ("model_T: \n{}".format(model_T))
                print ("model_R: \n{}".format(model_R))

            self.model_based_rl_cumulative_reward.append(cumulative_reward_episode)

        #all episodes have finished
        assert len(self.model_based_rl_cumulative_reward) == nEpisodes, "Length of reward list != nEpisodes " + repr(len(self.model_based_rl_cumulative_reward)) + repr(nEpisodes)
        if self.debug:
            print ("N_doubles: \n{}".format(N_doubles))
            print ("N_triples: \n{}".format(N_triples))
            print ("model_T: \n{}".format(model_T))
            print ("model_R: \n{}".format(model_R))

        return [V,policy]

    def epsilonGreedyBandit(self,nIterations, decay_epsilon=False):
        '''Epsilon greedy algorithm for bandits (assume no discount factor)

        Inputs:
        nIterations -- # of arms that are pulled
        decay_epsilon -- use a decaying value of epsilon = 1/iterationID

        Outputs:
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        # empiricalMeans = np.zeros(self.mdp.nStates)

        self.epsilon_greedy_reward = []
        #using a default value of epsilon in case decay_epsilon is False
        epsilon = 0.3
        #make sure we are using MDP with only one state
        assert self.mdp.nStates == 1, "We can only have 1 state in bandits problem. Given: " + repr(self.mdp.nStates)
        #initialize variables
        state = 0    #state is fixed at 0
        empiricalMeans_per_action = np.zeros([self.mdp.nActions])
        V_total = np.zeros([self.mdp.nStates])
        empiricalMeans = np.zeros([self.mdp.nStates])
        N = np.zeros([self.mdp.nActions])
        for iterationID in range(1, nIterations+1):
            if decay_epsilon:
                epsilon = 1./iterationID
            if epsilon > 0. and random.random() <= epsilon:
                action = random.randint(0, self.mdp.nActions-1)
                if self.debug:
                    print ("[exploration] action: {}".format(action))
                    print ("self.mdp.nActions: {}".format(self.mdp.nActions))
            else:
                action = np.argmax(empiricalMeans_per_action)
                if self.debug:
                    print ("[exploitation] action: {}".format(action))
                    print ("self.mdp.nActions: {}".format(self.mdp.nActions))
            #sample reward for this action
            reward,_ = self.sampleRewardAndNextState(state, action) #TODO check is this correct?
            V_total += reward
            self.epsilon_greedy_reward.append(reward)
            N[action] += 1
            empiricalMeans_per_action[action] = empiricalMeans_per_action[action] + (1./N[action])*(reward - empiricalMeans_per_action[action])

        assert len(self.epsilon_greedy_reward) == nIterations, "reward list doesn't match in size to nIterations"
        empiricalMeans = np.divide(V_total, nIterations)
        if self.debug:
            print("empiricalMeans_per_action: {}".format(empiricalMeans_per_action))
            print ("empiricalMeans: {}".format(empiricalMeans))

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

        self.thompson_sampling_reward = []
        #make sure we are using MDP with only one state
        assert self.mdp.nStates == 1, "We can only have 1 state in bandits problem. Given: " + repr(self.mdp.nStates)
        #initialize variables
        state = 0    #state is fixed at 0
        alpha = np.ones([self.mdp.nActions])
        beta = np.ones([self.mdp.nActions])
        #use the initial prior
        for action_idx in range(self.mdp.nActions):
            alpha[action_idx] = prior[action_idx, 0]
            beta[action_idx] = prior[action_idx, 1]
        samples = np.zeros([self.mdp.nActions, k])
        mean_reward = np.zeros([self.mdp.nActions])
        V_reward = np.zeros([self.mdp.nActions])
        Na = np.zeros([self.mdp.nActions])
        V_total = np.zeros([self.mdp.nStates])
        empiricalMeans = np.zeros([self.mdp.nStates])
        for iteration in range(nIterations):
            #generate samples for all actions from prior beta distribution
            for action_idx in range(self.mdp.nActions):
                samples[action_idx, :] = np.random.beta(alpha[action_idx], beta[action_idx], k)
                mean_reward[action_idx] = np.mean(samples[action_idx,:])
            action = np.argmax(mean_reward)
            reward,_ = self.sampleRewardAndNextState(state, action) #TODO check is this correct?
            if self.debug:
                print ("action: {}, reward: {}".format(action, reward))
                print ("hoeffding_term: {}".format(hoeffding_term))
            V_reward[action] += reward
            self.thompson_sampling_reward.append(reward)
            Na[action] += 1
            V_total += reward
            #update posterior beta distribution
            if reward == 1: alpha[action] += 1
            else: beta[action] += 1

        assert len(self.thompson_sampling_reward) == nIterations, "reward list doesn't match in size to nIterations"
        empiricalMeans_per_action = np.divide(V_reward, Na)
        empiricalMeans = np.divide(V_total, nIterations)
        if self.debug:
            print("empiricalMeans_per_action: {}".format(empiricalMeans_per_action))
            print ("empiricalMeans: {}".format(empiricalMeans))

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

        self.ucb_bandit_reward = []
        #make sure we are using MDP with only one state
        assert self.mdp.nStates == 1, "We can only have 1 state in bandits problem. Given: " + repr(self.mdp.nStates)
        #initialize variables
        state = 0    #state is fixed at 0
        #TODO start the value of emperical means at a high value so that all actions get a chance to be selected
        #keep it at one because the Na is also initialized to 1, this way we can say that the
        #initial reward was 1 and not 0
        #TODO or should we just change this and run all actions once and start off the initial estimate
        V_total = np.ones([self.mdp.nStates])
        empiricalMeans = np.ones([self.mdp.nStates])
        empiricalMeans_per_action = np.ones([self.mdp.nActions])
        hoeffding_term = np.ones([self.mdp.nActions])
        #TODO starting the count at 1 will still give the same results as the argmax
        #will choose the action based on which one is updated
        Na = np.ones([self.mdp.nActions])
        N = self.mdp.nActions
        for iteration in range(nIterations):
            for action_idx in range(self.mdp.nActions):
                hoeffding_term[action_idx] = empiricalMeans_per_action[action_idx] + math.sqrt(2*math.log(N)/Na[action_idx])
            action = np.argmax(hoeffding_term)
            reward,_ = self.sampleRewardAndNextState(state, action) #TODO check is this correct?
            V_total += reward
            self.ucb_bandit_reward.append(reward)
            # if self.debug:
            #     print ("action: {}, reward: {}".format(action, reward))
            #     print ("hoeffding_term: {}".format(hoeffding_term))
            Na[action] += 1
            N += 1
            empiricalMeans_per_action[action] = empiricalMeans_per_action[action] + (1./Na[action])*(reward - empiricalMeans_per_action[action])

        empiricalMeans = np.divide(V_total, nIterations)
        assert len(self.ucb_bandit_reward) == nIterations, "reward list doesn't match in size to nIterations"
        if self.debug:
            print("empiricalMeans_per_action: {}".format(empiricalMeans_per_action))
            print ("empiricalMeans: {}".format(empiricalMeans))

        return empiricalMeans

    def reinforce(self,s0,initialPolicyParams,nEpisodes,nSteps, naive_decay_lr=False, lr_constant=0.05, constant_lr=0.0, use_mc_est=False, epsilon=0.0, epsilon_decay=False, use_initial_pp=True, trip_assert=True, upd_rule=1, with_baseline=False, optionlr=0):
        '''reinforce algorithm.  Learn a stochastic policy of the form
        pi(a|s) = exp(policyParams(a,s))/[sum_a' exp(policyParams(a',s))]).
        This function should call the function sampleSoftmaxPolicy(policyParams,state) to select actions

        Inputs:
        s0 -- initial state
        initialPolicyParams -- parameters of the initial policy (array of |A|x|S| entries)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0)
        nSteps -- # of steps per episode
        naive_decay_lr -- decay lr using the formula 1/N[a,s] #TODO remove?  -- naive lr didn't help even with PP=[100]
        lr_constant -- multiply with this starting value of lr
        constant_lr -- (float) use a constant value of learning rate (takes presedence over decaying lr) #TODO remove?
        use_mc_est -- use Monte Carlo estimate instead of Gn #TODO remove?
        epsilon -- (0.3) add epsilon-greedy to improve performance of reinforce algorithm #TODO remove?
        epsilon_decay -- decay epsilon value linearly #TODO remove?
        use_initial_pp -- use the initial policy parameters provided #TODO remove?
        trip_assert -- #TODO remove?
        upd_rule -- which rule to follow during update? #TODO remove
                    1. update all actions for given state based on the Gn estimate
                    2. update only visited action, state pair
                    3. update all actions for given state using full jacobian and V_n(:, s_n)
                        NOTE: this also sets use_mc_est=True
        with_baseline -- use V as baseline

        Outputs:
        policyParams -- parameters of the final policy (array of |A|x|S| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        # policyParams = np.zeros((self.mdp.nActions,self.mdp.nStates))

        #setup variables
        self.reinforce_cumulative_reward = []
        if use_initial_pp:
            #TODO multiplying and adding 100 to initial random policy doesn't help
            policyParams = initialPolicyParams
        else:
            #TODO starting from constant policy definately helps
            policyParams = 100.0 * np.ones((self.mdp.nActions,self.mdp.nStates)) #TODO remove and uncomment above
        lr = 1
        N = np.zeros([self.mdp.nActions, self.mdp.nStates])
        policy = np.zeros([self.mdp.nActions, self.mdp.nStates])
        V_est = np.zeros([self.mdp.nActions, self.mdp.nStates])
        N_V_est = np.zeros([self.mdp.nActions, self.mdp.nStates])
        V_est2 = np.zeros([self.mdp.nActions, self.mdp.nStates])
        N_V_est2 = np.zeros([self.mdp.nActions, self.mdp.nStates])
        V_est_state = np.zeros([self.mdp.nStates])
        #execute all episodes
        for episode_idx in range(nEpisodes):
            #initialize start state
            state = s0
            #format for episode_path: (state, action, reward)
            episode_path = np.zeros([nSteps, 3])
            #loop through the entire episode and generate trajectory
            cumulative_reward_episode = 0.
            for step_idx in range(nSteps):
                #select an action based on epsilon (exploration vs exploitation)
                if epsilon_decay:
                    epsilon = (nEpisodes-episode_idx)/(4.*nEpisodes)
                if epsilon > 0. and random.random() <= epsilon:
                    action = random.randint(0, self.mdp.nActions-1)
                else:
                    action = self.sampleSoftmaxPolicy(policyParams, state)
                reward, state_p = self.sampleRewardAndNextState(state, action)
                episode_path[step_idx,:] = np.array([state,action,reward])
                if self.debug:
                    print ("[REINFORCE] reward: {}".format(reward))
                    print ("[REINFORCE] cumulative_reward: {}".format(reward*(self.mdp.discount**episode_step)))
                cumulative_reward_episode += reward*(self.mdp.discount**step_idx)
                state = state_p
            self.reinforce_cumulative_reward.append(cumulative_reward_episode)

            # #TODO use mc estimate when using 3rd update rule
            # if upd_rule == 3:
            #     use_mc_est = True

            if upd_rule == 3 or with_baseline:
                #compute visited states
                visited_states = np.unique(episode_path[:,0]).astype(int)
                debug_total_updates = 0
                for visit_state in visited_states:
                    # state_idx = episode_path[:,0].tolist().index(visit_state)
                    state_indicies = np.where(episode_path[:,0].astype(int) == visit_state)[0]
                    for state_idx in state_indicies:
                        visit_action = int(episode_path[state_idx, 1])
                        Gn_scalar = 0.
                        for idx in range(state_idx, nSteps):
                            Gn_scalar += episode_path[idx, 2]*(self.mdp.discount**(idx-state_idx))
                        assert Gn_scalar != 0., "Gn_scalar is 0. even after update"
                        N_V_est[visit_action, visit_state] += 1
                        #compute new average based on Monte Carlo estimate from lecture 3b slide 11
                        V_est[visit_action, visit_state] = V_est[visit_action, visit_state] + \
                            (1./N_V_est[visit_action, visit_state])*(Gn_scalar-V_est[visit_action, visit_state])
                        assert V_est[visit_action, visit_state] != 0., "V_est can't be 0. after update " + repr(Gn_scalar)
                        debug_total_updates += 1
                assert debug_total_updates == nSteps, "total number of updates don't equal nSteps " + repr(debug_total_updates)
            if with_baseline:
                #compute baseline for each state value function
                for idx_state in range(self.mdp.nStates):
                    V_est_state[idx_state] = np.mean(V_est[:,idx_state])
                #assert np.nonzero(V_est_state) == len(V_est_state), "V_est_state contains 0: " + repr(V_est_state)

            #loop through the entire episode and evaluate the policy and update policy parameters
            for step_idx in range(nSteps):
                state = int(episode_path[step_idx, 0])
                action = int(episode_path[step_idx, 1])
                #learning rate is based on (action, state). However, this learning
                #rate will vary for different actions of the same state's udpate. To keep
                #a constant learning rate for all actions of a given state, we use the current
                #count of (action, state) as the learning rate for all actions for this step
                N[action, state] += 1
                lr = 1./N[action, state]
                if not naive_decay_lr:
                    #learning rate decay schedule
                    lr = lr_constant*int((nEpisodes-episode_idx)/(nEpisodes/10)+1)
                if constant_lr > 0.0:
                    lr = constant_lr
                #TODO hard-coded lr
                if optionlr == 1:
                    if episode_idx < 60: lr=0.005
                    elif episode_idx < 100: lr=0.004
                    else: lr=0.003
                elif optionlr == 2:
                    if episode_idx < 60: lr=0.004
                    elif episode_idx < 100: lr=0.003
                    else: lr=0.002
                elif optionlr == 3:
                    if episode_idx < 60: lr=0.003
                    elif episode_idx < 100: lr=0.002
                    else: lr=0.001

                #compute Gn estimate for update
                Gn_upd = 0.
                #based on the episode trajectory generated, update the value estimates of the states visited
                if use_mc_est:
                    #compute the Gn based on the cumulative reward till the end of episode starting from this state
                    Gn_scalar = 0.
                    for idx in range(step_idx, nSteps):
                        reward = episode_path[idx, 2]
                        Gn_scalar += reward*(self.mdp.discount**(idx-step_idx))
                    #compute new average based on Monte Carlo estimate from lecture 3b slide 11
                    N_V_est2[action, state] += 1
                    V_est2[action, state] = V_est2[action, state] + \
                        (1./N_V_est2[action, state])*(Gn_scalar-V_est2[action, state])
                    Gn_upd = V_est2[action, state]
                else:
                    Gn_scalar = 0.
                    #compute Gn based on the cumulative reward till the end of episode starting from this state
                    for idx in range(step_idx, nSteps):
                        reward = episode_path[idx, 2]
                        Gn_scalar += (self.mdp.discount**(idx-step_idx))*reward
                    Gn_upd = Gn_scalar
                #TODO remove below code
                # if with_baseline:
                #     Gn_upd = Gn_upd - V_est_state[state]

                if upd_rule == 1:
                    #update the policy parameters using the update equation
                    #\theta <- \theta + \alpha*\gamma^n*Gn*\nabla(log \pi_\theta (a_n|s_n) )
                    #the derivative of log of softmax function is:
                    #   1-softmax(i) when i=j (diagonal entries in jacobian)
                    #   -softmax(j) when i!=j (all other entries in jacobian)
                    #so we don't need to compute the log and derivative, we only need to compute the softmax for each index
                    # get the policy parameters for this state as that is all that will get updated
                    shifted_policyParams = policyParams[:, state] - np.max(policyParams[:, state])
                    pp_state = shifted_policyParams
                    #initialize jacobian to |A|
                    jacobian = np.zeros([self.mdp.nActions])
                    for idx in range(jacobian.shape[0]):
                        if idx == action:
                            #jacobian[idx] = 1 - np.exp(pp_state[idx])/np.sum(np.exp(pp_state)) #TODO changing the softmax used
                            jacobian[idx] = 1 - np.exp(pp_state[action])/np.sum(np.exp(pp_state))
                        else:
                            #jacobian[idx] = 0 - np.exp(pp_state[idx])/np.sum(np.exp(pp_state)) #TODO changing the softmax used
                            jacobian[idx] = 0 - np.exp(pp_state[action])/np.sum(np.exp(pp_state))
                    # #TODO remove assert below as it trips a lot because for small number exponents its considered 0
                    # if state != 9 and trip_assert:
                    #     assert np.count_nonzero(jacobian) >= (len(jacobian)-1), "Zeros present in jacobian for REINFORCE (action, state): (" +\
                    #         repr(action) + ", " + repr(state) + ")\njacobian: " +\
                    #         repr(jacobian) + "\nPolicyParams: " + repr(pp_state) + "\nExp(PolicyParams): " + repr(np.exp(pp_state))
                    assert np.count_nonzero(np.isnan(jacobian)) == 0, "NaN present in jacobian for REINFORCE (action, state): (" +\
                        repr(action) + ", " + repr(state) + ")\njacobian: " +\
                        repr(jacobian) + "\nPolicyParams: " + repr(pp_state) + "\nExp(PolicyParams): " + repr(np.exp(pp_state))
                    #update the policy parameters
                    update_term = np.multiply(jacobian, lr*(self.mdp.discount**step_idx)*Gn_upd)
                    # print ("\nbefore:--------------------------\n{}".format(policyParams[:, state])) #TODO remove
                    policyParams[:, state] = policyParams[:, state] + update_term
                    #TODO try -update term
                    # print ("after:--------------------------\n{}\n".format(policyParams[:, state])) #TODO remove
                    #TODO remove following block
                    # if (state==5 and action==1) or (state==8 and action==3) or (state==13 and action==0) or (state==10 and action==2):
                    #     print ("\n------------ state: {} -- action: {} -- Gn: {} ---------------".format(state, action, Gn_upd))
                    #     # np.set_printoptions(precision=2, suppress=True)
                    #     # if Gn_upd >= 0.0:
                    #     #     #sometimes this can be positive as the transition is probabilities and can transition to other state than 9
                    #     #     #this can't happen when reward is -100 for pit
                    #     #     print ("remaining trajectory (rounded rewards): \n{}".format(np.round_(episode_path[step_idx:,:],decimals=2)))
                    #     #     for idx in range(step_idx, nSteps):
                    #     #         reward = episode_path[idx, 2]
                    #     #         print ("value at idx {}: {}".format(idx, (self.mdp.discount**(idx-step_idx))*reward))
                    #     print ("pp_state: {}".format(pp_state))
                    #     print ("jacobian: {}".format(jacobian))
                    #     print ("update_term: {}".format(update_term))
                    #     print ("updated pp_state: {}".format(policyParams[:, state]))
                elif upd_rule == 2:
                    #only update the visited action, state
                    pp_state = policyParams[:, state] - np.max(policyParams[:, state])
                    jacobian = 1.0 - (math.exp(pp_state[action])/(np.sum(np.exp(pp_state))))
                    assert np.count_nonzero(np.isnan(jacobian)) == 0, "NaN present in jacobian for REINFORCE (action, state): (" +\
                        repr(action) + ", " + repr(state) + ")\njacobian: " +\
                        repr(jacobian) + "\nPolicyParams: " + repr(pp_state) + "\nExp(PolicyParams): " + repr(np.exp(pp_state))
                    update_term = jacobian*lr*(self.mdp.discount**step_idx)*Gn_upd
                    policyParams[action, state] += update_term
                elif upd_rule == 3:
                    pp_state = policyParams[:, state] - np.max(policyParams[:, state])
                    #compute full jacobian
                    jacobian = np.zeros([self.mdp.nActions, self.mdp.nActions])
                    for col_idx in range(self.mdp.nActions):
                        for row_idx in range(self.mdp.nActions):
                            if col_idx == row_idx:
                                value = 1.0 - (math.exp(pp_state[col_idx])/np.sum(np.exp(pp_state)))
                            else:
                                value = 0.0 - (math.exp(pp_state[col_idx])/np.sum(np.exp(pp_state)))
                            jacobian[row_idx, col_idx] = value
                    assert np.count_nonzero(np.isnan(jacobian)) == 0, "NaN present in jacobian for REINFORCE (action, state): (" +\
                        repr(action) + ", " + repr(state) + ")\njacobian: " +\
                        repr(jacobian) + "\nPolicyParams: " + repr(pp_state) + "\nExp(PolicyParams): " + repr(np.exp(pp_state))
                    value_state = V_est[:, state]
                    update_term = (self.mdp.discount**step_idx)*lr*np.matmul(jacobian, np.transpose(value_state))
                    policyParams[:, state] += update_term

            # if episode_idx % 100 == 0:
            if self.debug and episode_idx % 100 == 0:
                print ("episode_idx: {}".format(episode_idx))
                # print ("episode_path: \n{}".format(episode_path))
                print ("policyParams[:,state]: \n{}".format(policyParams[:, state]))
                print ("Gn_upd: \n{}".format(Gn_upd))
                print ("jacobian {}".format(jacobian))
                print ("update_term for action,state({},{}): {}".format(action, state, update_term))

        assert len(self.reinforce_cumulative_reward) == nEpisodes, "Length of reward list != nEpisodes " + repr(len(self.reinforce_cumulative_reward)) + repr(nEpisodes)
        #compute the policy from the policy parameters
        for state in range(self.mdp.nStates):
            # TODO check if it works first and then: shifted_policyParams = policyParams[:, state] - np.max(policyParams[:, state])
            shifted_policyParams = policyParams[:, state] - np.max(policyParams[:, state])
            pp_state = shifted_policyParams
            policy[:, state] = np.exp(pp_state)/np.sum(np.exp(pp_state))
        assert np.count_nonzero(np.isnan(policy)) == 0, "NaN present in final policy for REINFORCE \npolicy: " +\
            repr(policy) + "\nPolicyParams: " + repr(policyParams) + "\nExp(PolicyParams): " + repr(np.exp(policyParams))

        return policyParams, policy

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
        iterId = 0
        policy = np.zeros(self.mdp.nStates,int)
        self.q_learning_cumulative_reward = []
        #Q-learning convergence requires us to visit every state inifinite number of times
        #thus it is better to run it for specified episodes*steps unless some other convergence condition is provided
        # like in DQN the convergence requires the moving average of rewards to be >= 200
        for episode_idx in range(nEpisodes):
            iterId = 0
            cumulative_reward_episode = 0.
            for episode_step in range(nSteps):
                action_chosen = False
                #select an action based on epsilon (exploration vs exploitation)
                if epsilon > 0. and random.random() <= epsilon:
                    #explore with probability epsilon
                    action = random.randint(0, self.mdp.nActions-1)
                    action_chosen = True
                    if self.debug:
                        print ("[exploration] action: {}".format(action))
                        print ("self.mdp.nActions: {}".format(self.mdp.nActions))
                if (not action_chosen) and temperature > 0:
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
                #update Q-value for current state,action pair
                Q_new = copy.deepcopy(Q)
                #action_p is the best action from state_p under this Q function
                action_p = np.argmax(Q[:, state_p], axis=0)
                Q_new[action, state]= Q[action, state] + lr*(reward+(self.mdp.discount*Q[action_p, state_p])-Q[action, state])
                if self.debug:
                    print ("reward: {}".format(reward))
                    print ("cumulative_reward: {}".format(reward*(self.mdp.discount**episode_step)))
                cumulative_reward_episode += reward*(self.mdp.discount**episode_step)
                state = state_p
                Q = Q_new
            #episode has finished, reset starting state for next episode
            state = s0
            self.q_learning_cumulative_reward.append(cumulative_reward_episode)
            if self.debug:
                print ("\n-------------------- episode_idx: {} ------------------------".format(episode_idx))
                print ("N: {}".format(N))
                print ("Q: {}".format(Q))
                print ("policy: {}".format(np.argmax(Q, axis=0)))
                print ("lr: {}".format(lr))
                print ("update: {}".format(Q_new - Q))
                print ("cumulative_reward: {}".format(cumulative_reward_episode))
        #all episodes have finished
        assert len(self.q_learning_cumulative_reward) == nEpisodes, "Length of reward list != nEpisodes " + repr(len(self.q_learning_cumulative_reward)) + repr(nEpisodes)
        #update the policy using the final Q-function
        policy = np.argmax(Q, axis=0)

        #sanity check on output
        assert Q.ndim == 2, "Invalid Final Q: Wrong dimensionality " + repr(Q.ndim)
        assert Q.shape[0] == self.mdp.nActions and Q.shape[1] == self.mdp.nStates, "Invalid Final Q: Wrong shape " + repr(Q.shape)
        assert policy.ndim == 1, "Invalid Final policy: Wrong dimensionality " + repr(policy.ndim)
        assert policy.shape[0] == self.mdp.nStates, "Invalid Final policy: Wrong shape " + repr(policy.shape)

        return [Q,policy]

    def get_q_learning_cumulative_reward(self):
        '''Keeping the signature of the q_learning function unchanged, I have added this
        seperate function to keep track of the cumulative_reward per episode generated from
        the latest run of q-learning
        '''
        return np.array(self.q_learning_cumulative_reward)

    def get_reinforce_cumulative_reward(self):
        '''Keeping the signature of the reinforce function unchanged, I have added this
        seperate function to keep track of the reinforce_cumulative_reward per episode generated from
        the latest run of reinforce algorithm
        '''
        return np.array(self.reinforce_cumulative_reward)

    def get_model_based_rl_cumulative_reward(self):
        '''Keeping the signature of the modelBasedRL function unchanged, I have added this
        seperate function to keep track of the model_based_rl_cumulative_reward per episode generated from
        the latest run of modelBasedRL
        '''
        return np.array(self.model_based_rl_cumulative_reward)

    def get_epsilon_greedy_reward(self):
        '''Keeping the signature of the epsilonGreedyBandit function unchanged, I have added this
        seperate function to keep track of the epsilon_greedy_reward per iteration generated from
        the latest run of epsilonGreedyBandit
        '''
        return np.array(self.epsilon_greedy_reward)

    def get_ucb_bandit_reward(self):
        '''Keeping the signature of the UCBbandit function unchanged, I have added this
        seperate function to keep track of the ucb_bandit_reward per iteration generated from
        the latest run of UCBbandit
        '''
        return np.array(self.ucb_bandit_reward)

    def get_thompson_sampling_reward(self):
        '''Keeping the signature of the thompsonSamplingBandit function unchanged, I have added this
        seperate function to keep track of the thompson_sampling_reward per iteration generated from
        the latest run of thompsonSamplingBandit
        '''
        return np.array(self.thompson_sampling_reward)


# # we are using Monte Carlo estimate instead of only using Gn computed in this step
# if first_visit_mce:
#     visited_states = np.unique(episode_path[:,0]).astype(int)
#     for visit_state in visited_states:
#         state_idx = episode_path[:,0].tolist().index(visit_state)
#         visit_action = int(episode_path[state_idx, 1])
#         Gn_scalar = 0.
#         for idx in range(state_idx, nSteps):
#             Gn_scalar += episode_path[idx, 2]*(self.mdp.discount**(idx-state_idx))
#         assert Gn_scalar != 0., "Gn_scalar is 0. even after update"
#         N_V_est[visit_action, visit_state] += 1
#         #compute new average based on Monte Carlo estimate from lecture 3b slide 11
#         V_est[visit_action, visit_state] = V_est[visit_action, visit_state] + \
#             (1./N_V_est[visit_action, visit_state])*(Gn_scalar-V_est[visit_action, visit_state])
#         assert V_est[visit_action, visit_state] != 0., "V_est can't be 0. after update " + repr(Gn_scalar)
