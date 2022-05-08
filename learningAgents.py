import random,util,time
import numpy as np
import math

from blackBox import blackBox


class ReinforcementAgent(object):
    def __init__(self, alpha=1.0, epsilon=0.05, gamma=0.8, numTraining = 10):
        """
        Sets options, which can be passed in via the Pacman command line using -a alpha=0.5,...
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(gamma)
        self.numTraining = int(numTraining)
        
        self.totalReward = 0
        
    def testinit(self):
        self.totalReward = 0

    def getLegalActions(self,state):
        """
          Get the actions available for a given
          state. This is what you should use to
          obtain legal actions for a state
        """
        return ["change","nochange"]

    def observeTransition(self, state,action,nextState,deltaReward):
        """
            Called by environment to inform agent that a transition has
            been observed. This will result in a call to self.update
            on the same arguments

            NOTE: Do *not* override or call this function
        """
        self.episodeRewards += deltaReward
        self.update(state,action,nextState,deltaReward)

    def startEpisode(self):
        """
          Called by environment when new episode is starting
        """
        self.lastState = None
        self.lastAction = None
        self.episodeRewards = 0.0

    def stopEpisode(self):
        """
          Called by environment when episode is done
        """
        if self.episodesSoFar < self.numTraining:
            self.accumTrainRewards += self.episodeRewards
        else:
            self.accumTestRewards += self.episodeRewards
        self.episodesSoFar += 1
        if self.episodesSoFar >= self.numTraining:
            # Take off the training wheels
            self.epsilon = 0.0    # no exploration
            self.alpha = 0.0      # no learning

    def isInTraining(self):
        return self.episodesSoFar < self.numTraining

    def isInTesting(self):
        return not self.isInTraining()

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def setLearningRate(self, alpha):
        self.alpha = alpha

    def setDiscount(self, discount):
        self.discount = discount

    def doAction(self,state,action):
        """
            Called by inherited class when
            an action is taken in a state
        """
        self.lastState = state
        self.lastAction = action


    def observationFunction(self, state):
        """
            This is where we ended up after our last action.
            The simulation should somehow ensure this is called
        """
        if not self.lastState is None:
            reward = state.getScore() - self.lastState.getScore()
            self.observeTransition(self.lastState, self.lastAction, state, reward)
        return state

    def registerInitialState(self, state):
        self.startEpisode()
        if self.episodesSoFar == 0:
            print('Beginning %d episodes of Training' % (self.numTraining))

    def final(self):
        """
          Called by Pacman game at the terminal state
        """
        print("paras --------->")
        print("self.alpha")
        print(self.alpha)
        print("self.discount")
        print(self.discount)
        print("self.epsilon")
        print(self.epsilon)
        print("result ---->")
        print("self.totalReward")
        print(self.totalReward)

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)
        self.Q = {}
        self.totalReward = 0
        
        
    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        key = tuple([state, action])
        if not key in self.Q.keys():
            return 0
        return self.Q[key]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        legalActions = self.getLegalActions(state)
        if len(legalActions)==0:
            return 0
        action = None
        maxQ = -999999
        for tmp in legalActions:
            
            if self.getQValue(state, tmp) > maxQ:
                action = tmp
                maxQ = self.getQValue(state,action)
        return maxQ

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        legalActions = self.getLegalActions(state)
        if len(legalActions)==0:
            return None
        action = None
        maxQ = -999999
        print(legalActions)
        for tmp in legalActions:
            print(tmp + " Q-value:")
            print(self.getQValue(state, tmp))
            if self.getQValue(state, tmp) > maxQ:
                action = tmp
                maxQ = self.getQValue(state,action)
        return action
        
 
    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        if len(legalActions)==0:
            return None
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        else:
            return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        key = tuple([state,action])
        self.totalReward += reward
        self.Q[tuple([state,action])] = self.getQValue(state,action)+self.alpha*(
                    reward+self.discount*self.getValue(nextState)-self.getQValue(state,action))

    def getValue(self, state):
        return self.computeValueFromQValues(state)

class SARSAAgent(ReinforcementAgent):
    """
      SARSA Agent

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.Q = {}
        self.nextaction = None
        self.totalReward = 0
        
    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        key = tuple([state, action])
        if not key in self.Q.keys():
            return 0
        return self.Q[key]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        legalActions = self.getLegalActions(state)
        if len(legalActions)==0:
            return 0
        action = None
        maxQ = -999999
        for tmp in legalActions:
            if self.getQValue(state, tmp) > maxQ:
                action = tmp
                maxQ = self.getQValue(state,action)
        return maxQ

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        legalActions = self.getLegalActions(state)
        if len(legalActions)==0:
            return None
        action = None
        maxQ = -999999
        print(legalActions)
        for tmp in legalActions:
            print(tmp + " Q-value:")
            print(self.getQValue(state, tmp))
            if self.getQValue(state, tmp) > maxQ:
                action = tmp
                maxQ = self.getQValue(state,action)
        return action
        
    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action'
        if self.nextaction != None:
            return self.nextaction
            
        legalActions = self.getLegalActions(state)
        action = None
        if len(legalActions)==0:
            return None
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        else:
            return self.computeActionFromQValues(state)
    
    def getNextAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        if len(legalActions)==0:
            return None
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        else:
            return self.computeActionFromQValues(state)
            
    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        self.totalReward += reward
        self.nextaction = self.getNextAction(nextState)
        key = tuple([state,action])
        self.Q[tuple([state,action])] = self.getQValue(state,action)+self.alpha*(
                    reward+self.discount*self.getQValue(nextState, self.nextaction)-self.getQValue(state,action))

class ApproximateQAgent(QLearningAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)
        self.weights = {}
        self.weights['NCarNumber'] = random.random()
        self.weights['ECarNumber'] = random.random()
        self.weights['NCarWait'] = random.random()
        self.weights['ECarWait'] = random.random()
        self.weights['nextNCarNumber'] = random.random()
        self.weights['nextECarNumber'] = random.random()
        self.weights['nextNCarWait'] = random.random()
        self.weights['nextECarWait'] = random.random()
        self.weights['bias'] = random.random()
        self.totalReward = 0
        
    def getFeatures(self, state, action):
        features = {}
        # trafficlight condition
        NTrafficLight = state[2][0]
        ETrafficLight = state[2][1]
        # car number in different directions
        NCarNumber = state[0][0]
        ECarNumber = state[0][1]
        features['NCarNumber'] = NCarNumber * NTrafficLight
        features['ECarNumber'] = ECarNumber  * ETrafficLight
        # car waittime in different directions
        NCarWait = state[1][0]
        ECarWait = state[1][1]
        features['NCarWait'] = NCarWait * NTrafficLight 
        features['ECarWait'] = ECarWait * ETrafficLight
        
        # action
        if action == "nochange":
            nextNTrafficLight = NTrafficLight
            nextETrafficLight = ETrafficLight
        else:
            nextNTrafficLight = NTrafficLight * -1
            nextETrafficLight = ETrafficLight * -1
        
        if nextNTrafficLight == 1:
            nextNCarNumber = NCarNumber - 2 if NCarNumber > 1 else 0
        elif nextETrafficLight == 1:
            nextECarNumber = ECarNumber - 2 if ECarNumber > 1 else 0
         
        features['nextNCarWait'] = NCarWait * nextNTrafficLight
        features['nextECarWait'] = ECarWait * nextETrafficLight
        features['nextNCarNumber'] = NCarNumber * nextNTrafficLight
        features['nextECarNumber'] = ECarNumber * nextETrafficLight
        features['bias'] = 1
        
        return features


    def getWeights(self):
        return self.weights
        
    def weightsNormalize(self):
        res = 0
        for key in self.weights.keys():
            res += self.weights[key]
        for key in self.weights.keys():
            self.weights[key] = self.weights[key]/ (res + 1e-12)

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        res = 0
        features = self.getFeatures(state, action)
        weights = self.getWeights()

        for key in features.keys():
            res += weights[key] * features[key]
        return res

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        self.totalReward += reward
        features = self.getFeatures(state, action)
        print("=============")
        print(features)
        print("\n")
        print(self.weights)
        difference = (reward + self.discount * self.getValue(nextState)
                        ) - self.getQValue(state, action)
        for key in features.keys():
            self.weights[key] = self.weights[key] + self.alpha * difference * features[key]
        self.weightsNormalize()
        print("\n")
        print(self.weights)

class DeepQAgent(QLearningAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)
        self.blackBox = blackBox()
        self.recall_buffer = []
        self.totalReward = 0
    
    def getFeatures(self, state, action):
        if action == "noaction":
            actionnumber = 0
        else:
            actionnumber = 1
        x = [state[0][0], state[0][1], state[1][0], state[1][1], state[2][0], state[2][1], actionnumber]
        features = np.array(x)
        return features
        
    def getQValue(self, state, action):
        features = self.getFeatures(state, action)
        QValue = self.blackBox.predict(features)
        return QValue

    def update(self, state, action, nextState, reward):
        """
           Update the network
        """
        self.totalReward += reward
        self.recall_buffer += [[state, action, reward, nextState]]
        data_size = len(self.recall_buffer)
        batch_size = 20
        if data_size < batch_size:
            return
        
        idx = np.random.permutation(data_size)
        np_recall = np.array(self.recall_buffer)
        x = np_recall[idx]
        max_iters = int(data_size / batch_size)
        if max_iters > 20:
            max_iters = 20
       
        for iters in range(max_iters):

            batch_x = x[iters*batch_size:(iters+1)*batch_size]
            batch_y = np.array([])
            batch_xx = np.zeros([1,7])

            for i in range(len(batch_x)):
                batch_y = np.append(batch_y, batch_x[i][2] + self.getValue(batch_x[i][3]))

                batch_xx = np.insert(batch_xx, 0,  values = [self.getFeatures(batch_x[i][0], batch_x[i][1])], axis = 0)
            batch_xx = np.delete(batch_xx, -1, axis=0)

            self.blackBox.update(batch_xx, batch_y)



            

    
        