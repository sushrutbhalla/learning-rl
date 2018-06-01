import copy
import numpy as np
from numpy import linalg as LA

class MDP:
    '''A simple MDP class.  It includes the following members'''

    def __init__(self,T,R,discount):
        '''Constructor for the MDP class

        Inputs:
        T -- Transition function: |A| x |S| x |S'| array
        R -- Reward function: |A| x |S| array
        discount -- discount factor: scalar in [0,1)

        The constructor verifies that the inputs are valid and sets
        corresponding variables in a MDP object'''

        assert T.ndim == 3, "Invalid transition function: it should have 3 dimensions"
        self.nActions = T.shape[0]
        self.nStates = T.shape[1]
        assert T.shape == (self.nActions,self.nStates,self.nStates), "Invalid transition function: it has dimensionality " + repr(T.shape) + ", but it should be (nActions,nStates,nStates)"
        assert (abs(T.sum(2)-1) < 1e-5).all(), "Invalid transition function: some transition probability does not equal 1"
        self.T = T
        assert R.ndim == 2, "Invalid reward function: it should have 2 dimensions"
        assert R.shape == (self.nActions,self.nStates), "Invalid reward function: it has dimensionality " + repr(R.shape) + ", but it should be (nActions,nStates)"
        self.R = R
        assert 0 <= discount < 1, "Invalid discount factor: it should be in [0,1)"
        self.discount = discount
        self.debug = False

    def valueIteration(self,initialV,nIterations=np.inf,tolerance=0.01):
        '''Value iteration procedure
        V <-- max_a R^a + gamma T^a V

        Inputs:
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs:
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        assert initialV.ndim == 1, "Invalid initialV: it has dimensionality " + repr(initialV.ndim)
        assert initialV.shape[0] == self.nStates, "Invalid initialV shape: it has shape " + repr(initialV.shape)
        V_star = initialV  #using initialV as it was given in slide 15 (2b) that all initial estimates
                           #for value iteration will terminate (just with different number of iterations
        iterId = 0
        epsilon = 0.
        changeInV = True
        V_act = np.empty([self.nActions, self.nStates])
        while changeInV:
            for act_idx in range(self.nActions):
                if self.debug:
                    print ("R[act_idx].shape: {}".format(self.R[act_idx].shape))
                    print ("T[act_idx].shape: {}".format(self.T[act_idx].shape))
                    print ("V_star.shape: {}".format(V_star.shape))
                    print ("right term shape: {}".format(np.matmul(self.T[act_idx], V_star).shape))
                    print ("full term shape: {}".format((self.R[act_idx] + (self.discount * np.matmul(self.T[act_idx], V_star))).shape))
                #for each action, compute the V and then select V_star based on max of element wise
                V_act[act_idx] = self.R[act_idx] + (self.discount * np.matmul(self.T[act_idx], V_star))
            if self.debug:
                print ("V_act[0] max {}".format(np.amax(V_act, axis=0)))
                print ("V_star {}".format(V_star))
            iterId += 1
            epsilon = LA.norm(np.subtract(V_star,np.amax(V_act, axis=0)), np.inf)
            #print ("[DEBUG] epsilon: {}".format(epsilon))
            if (nIterations == np.inf and tolerance == 0. and epsilon == 0. ) or \
               (tolerance != 0. and epsilon <= tolerance) or \
               (nIterations != np.inf and iterId == nIterations):
                #the algorithm will only converge when the value function for states stops changing OR
                #the algorithm will converge when the change in value function is less than tolerance
                # (based on the comment on piazza that stop the algo when the epsilon <= tolerance) OR
                #the algorithm will converge when the nIterations have expired
                changeInV = False
            V_star = np.amax(V_act, axis=0)

        V = V_star
        if self.debug:
            print ("V: {}".format(V))
            print ("iterId: {}".format(iterId))
            print ("epsilon: {}".format(epsilon))

        return [V,iterId,epsilon]

    def extractPolicy(self,V):
        '''Procedure to extract a policy from a value function
        pi <-- argmax_a R^a + gamma T^a V

        Inputs:
        V -- Value function: array of |S| entries

        Output:
        policy -- Policy: array of |S| entries'''

        assert V.ndim == 1, "Invalid V: it has dimensionality " + repr(V.ndim)
        assert V.shape[0] == self.nStates, "Invalid V: it has shape " + repr(V.shape)

        #the policy can be extracted by running the value iteration step one more time
        #because the value iteration has converged, we should have a stationary policy, so
        #running the value iteration step one more time won't change the policy
        V_act = np.empty([self.nActions, self.nStates])
        for act_idx in range(self.nActions):
            if self.debug:
                print ("right term shape: {}".format(np.matmul(self.T[act_idx], V).shape))
                print ("full term shape: {}".format((self.R[act_idx] + (self.discount * np.matmul(self.T[act_idx], V))).shape))
            #for each action, compute the V and then select V based on max of element wise
            V_act[act_idx] = self.R[act_idx] + (self.discount * np.matmul(self.T[act_idx], V))
        if self.debug:
            #seems like 1 is 'S' (save) and 0 is 'A' (advertise)
            print ("V_act amax {}".format(np.amax(V_act, axis=0)))
            print ("V_act argmax {}".format(np.argmax(V_act, axis=0)))
        policy = np.argmax(V_act, axis=0)

        return policy

    def evaluatePolicy(self,policy):
        '''Evaluate a policy by solving a system of linear equations
        V^pi = R^pi + gamma T^pi V^pi

        Input:
        policy -- Policy: array of |S| entries

        Ouput:
        V -- Value function: array of |S| entries'''

        #get the reward signal and transition matrix for the current policy
        R_pi = np.empty([self.nStates])
        #TODO delete: T_pi = np.zeros([self.nActions,self.nStates,self.nStates], dtype=np.float)
        T_pi = np.zeros([self.nStates,self.nStates], dtype=np.float)
        V_pi = np.zeros([self.nStates])
        changeInV = True
        for state_idx in range(self.nStates):
            action = policy[state_idx]
            R_pi[state_idx] = self.R[action, state_idx]
            T_pi[state_idx] = self.T[action, state_idx]
        if self.debug:
            print ("policy: {}".format(policy))
            print ("R_pi: {}".format(R_pi))
            print ("T_pi: {}".format(T_pi))
            print ("R_pi shape: {}".format(R_pi.shape))
            print ("T_pi shape: {}".format(T_pi.shape))
            print ("T shape: {}".format(self.T.shape))
            print ("V_pi shape: {}".format(V_pi.shape))
            print ("right term: {}".format(np.matmul(T_pi, V_pi).shape))
        while changeInV:
            V_new = R_pi + (self.discount*np.matmul(T_pi, V_pi))
            if np.array_equal(V_new, V_pi):
                changeInV = False
            V_pi = V_new
        V = V_pi

        return V

    def policyIteration(self,initialPolicy,nIterations=np.inf):
        '''Policy iteration procedure: alternate between policy
        evaluation (solve V^pi = R^pi + gamma T^pi V^pi) and policy
        improvement (pi <-- argmax_a R^a + gamma T^a V^pi).

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        nIterations -- limit on # of iterations: scalar (default: inf)

        Outputs:
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        #sanity checking
        assert initialPolicy.ndim == 1, "Invalid initialPolicy: it has dimensionality " + repr(initialPolicy.ndim)
        assert initialPolicy.shape[0] == self.nStates, "Invalid initialPolicy: it has shape " + repr(initialPolicy.shape)
        # loop till the policy stops updates
        changeInP = True
        policy = initialPolicy
        V = np.zeros(self.nStates)
        iterId = 0
        while changeInP or (nIterations != np.inf and iterId < nIterations):
            V_eval = self.evaluatePolicy(policy)
            #generate new policy using V_eval
            V_act = np.empty([self.nActions, self.nStates])
            for act_idx in range(self.nActions):
                V_act[act_idx] = self.R[act_idx] + (self.discount * np.matmul(self.T[act_idx], V_eval))
            policy_new = np.argmax(V_act, axis=0)
            if self.debug:
                print ("V_eval: {}".format(V_eval))
                print ("policy_new: {}".format(policy_new))
            if np.array_equal(policy_new, policy) or np.array_equal(V_eval, V):
                #from lecture 3a video 14:00, we should also have stopping condition where the value function is same
                changeInP = False
            policy = policy_new
            V = V_eval
            iterId += 1

        #TODO the comments have epsilon
        return [policy,V,iterId]

    def evaluatePolicyPartially(self,policy,initialV,nIterations=np.inf,tolerance=0.01):
        '''Partial policy evaluation:
        Repeat V^pi <-- R^pi + gamma T^pi V^pi

        Inputs:
        policy -- Policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs:
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        #TODO check if this is the right way of doing it, as the equation at the top doesn't have a equal sign, it has a assignment
        #get the reward signal and transition matrix for the current policy
        R_pi = np.empty([self.nStates])
        #TODO delete: T_pi = np.zeros([self.nActions,self.nStates,self.nStates], dtype=np.float)
        T_pi = np.zeros([self.nStates,self.nStates], dtype=np.float)
        V_pi = initialV
        changeInV = True
        iterId = 0
        epsilon = 0.
        for state_idx in range(self.nStates):
            action = policy[state_idx]
            R_pi[state_idx] = self.R[action, state_idx]
            T_pi[state_idx] = self.T[action, state_idx]
        if self.debug:
            print ("policy: {}".format(policy))
            print ("R_pi: {}".format(R_pi))
            print ("T_pi: {}".format(T_pi))
            print ("R_pi shape: {}".format(R_pi.shape))
            print ("T_pi shape: {}".format(T_pi.shape))
            print ("T shape: {}".format(self.T.shape))
            print ("V_pi shape: {}".format(V_pi.shape))
            print ("right term: {}".format(np.matmul(T_pi, V_pi).shape))
        if nIterations == 0:
            changeInV = False
        while changeInV:
            V_new = R_pi + (self.discount*np.matmul(T_pi, V_pi))
            iterId += 1
            epsilon = LA.norm(np.subtract(V_new,V_pi), np.inf)
            if (nIterations == np.inf and tolerance == 0. and epsilon == 0. ) or \
               (tolerance != 0. and epsilon <= tolerance) or \
               (nIterations != np.inf and iterId == nIterations):
                changeInV = False
            V_pi = V_new
        V = V_pi

        return [V,iterId,epsilon]

    def modifiedPolicyIteration(self,initialPolicy,initialV,nEvalIterations=5,nIterations=np.inf,tolerance=0.01):
        '''Modified policy iteration procedure: alternate between
        partial policy evaluation (repeat a few times V^pi <-- R^pi + gamma T^pi V^pi)
        and policy improvement (pi <-- argmax_a R^a + gamma T^a V^pi)

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nEvalIterations -- limit on # of iterations to be performed in each partial policy evaluation: scalar (default: 5)
        nIterations -- limit on # of iterations to be performed in modified policy iteration: scalar (default: inf)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs:
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        # # temporary values to ensure that the code compiles until this
        # # function is coded
        # policy = np.zeros(self.nStates)
        # V = np.zeros(self.nStates)
        # iterId = 0
        # epsilon = 0

        #sanity checking
        assert initialPolicy.ndim == 1, "Invalid initialPolicy: it has dimensionality " + repr(initialPolicy.ndim)
        assert initialPolicy.shape[0] == self.nStates, "Invalid initialPolicy: it has shape " + repr(initialPolicy.shape)
        assert initialV.ndim == 1, "Invalid initialV: it has dimensionality " + repr(initialV.ndim)
        assert initialV.shape[0] == self.nStates, "Invalid initialV shape: it has shape " + repr(initialV.shape)
        # loop till the policy stops updates or we reach tolerance
        changeInP = True
        policy = initialPolicy
        V = initialV
        V_next = initialV
        iterId = 0
        epsilon = 0.
        while changeInP or (nIterations != np.inf and iterId < nIterations):
            V_eval, _, _ = self.evaluatePolicyPartially(policy, V_next, nIterations=nEvalIterations, tolerance=tolerance)
            #TODO do I worry about the epsilon from the previous call? is it the same epsilon as the one computed below?
            #generate new policy using V_eval
            V_act = np.empty([self.nActions, self.nStates])
            for act_idx in range(self.nActions):
                V_act[act_idx] = self.R[act_idx] + (self.discount * np.matmul(self.T[act_idx], V_eval))
            policy_new = np.argmax(V_act, axis=0)
            if self.debug:
                print ("V_eval: {}".format(V_eval))
                print ("policy_new: {}".format(policy_new))
            #compute V^(n+1) TODO the following lines seem useless
            V_act2 = np.empty([self.nActions, self.nStates])
            for act_idx in range(self.nActions):
                #for each action, compute the V and then select V_next based on max of element wise
                V_act2[act_idx] = self.R[act_idx] + (self.discount * np.matmul(self.T[act_idx], V_eval))
            V_next = np.amax(V_act2, axis=0)
            iterId += 1
            epsilon = LA.norm(np.subtract(V_eval, V_next), np.inf)
            if np.array_equal(policy_new, policy) or np.array_equal(V_eval, V_next) or (nIterations != np.inf and epsilon <= tolerance) or (nIterations == iterId):
                #from lecture 3a video 14:00, we should also have stopping condition where the value function is same
                changeInP = False
            policy = policy_new
            V = V_eval


        return [policy,V,iterId,epsilon]

