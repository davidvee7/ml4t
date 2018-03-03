
import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=20, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):
        #from professor
        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0

        #from me
        self.q = np.random.uniform(-1.0,1.0,(num_states,num_actions))
        #Alpha is the learning rate, which falls between 0 and 1 inclusively.  A learning rate of 1 makes the learner only
        # use the most recent info, while a  learning rate of 0 makes the learner forget everything.
        self.alpha = alpha
        #Gamma is the importance of future rewards.  Towards 0 is short sightedness, while towards 1 is long sightedness
        self.gamma = gamma
        #How often a random action will be chosen.
        self.RandomActionRate = rar
        #How quickly the learner will slow down or stop doing random actions.
        self.RandomActionDecayRate = radr
        self.dyna = dyna

        self.R = np.ones((num_states,num_actions))

        self.experienceTuples = []

        self.num_states = num_states
    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        action = rand.randint(0, self.num_actions-1)
        if self.verbose: print "s =", s,"a =",action
        maximums = np.argmax(self.q,axis=1)

        someNum = np.random.uniform()
        if someNum < (self.RandomActionRate):
            # print "do something random"
            self.a= action
            return action
        else:
            action = maximums[s]
        self.a= action

        return action

# query is the core method of the Q-Learner. It should keep track of the last state s and the last
    # action a, then use the new information s_prime and r to update the Q table.
    # The learning instance, or experience tuple is <s, a, s_prime, r>. query() should return an
    # integer, which is the next action to take. Note that it should choose a random action with
    # probability rar, and that it should update rar according to the decay rate radr at each step.
    #  Details on the arguments:
#     s_prime integer, the the new state.
#     r float, a real valued immediate reward.
    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        #from proessor

        lastAction = self.a
        lastState = self.s
        action = rand.randint(0, self.num_actions-1)

        if self.verbose: print "s =", s_prime,"a =",action,"r =",r

        someNum = np.random.uniform()
        maximums = np.argmax(self.q,axis=1)

        if someNum < (self.RandomActionRate):
            pass #because action is already set to something random
        else:
            action = maximums[s_prime]

        #Decay the RandomActionRate.
        self.RandomActionRate = self.RandomActionRate*self.RandomActionDecayRate

        #(1-self.alpha)*self.q[self.s,self.a] == how much you want previous experiences to count
        #self.alpha*(r+ self.gamma*self.q[s_prime,maximums[s_prime]]) == how much this reward and future rewards will get you
        self.q[self.s,self.a]= (1-self.alpha)*self.q[self.s,self.a] + \
            self.alpha*(r+ self.gamma*self.q[s_prime,maximums[s_prime]])
        self.a= action
        self.s= s_prime

        #So the q learner can remember what rewards are associated with taken a given action in a given state
        self.experienceTuples.append((lastState,lastAction,s_prime,r))

        self.R[lastState,lastAction] = (1-self.alpha) * self.R[lastState,lastAction] + self.alpha*r

        for i in range(self.dyna):

            #alternative method
            luckyNumber = np.random.choice(len(self.experienceTuples))
            drawnExperience=self.experienceTuples[luckyNumber]
            r = self.R[drawnExperience[0],drawnExperience[1]]
            self.q[drawnExperience[0],drawnExperience[1]] = (1-self.alpha)*self.q[drawnExperience[0],drawnExperience[1]] + \
            self.alpha*(r+ self.gamma*self.q[drawnExperience[2],maximums[drawnExperience[2]]])

        return action

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"