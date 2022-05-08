from mapAgent import mapAgent
from learningAgents import QLearningAgent, ApproximateQAgent, SARSAAgent, DeepQAgent
from mapDisplay import mapDisplay

class modelController(object):
    def __init__(self, modelChoose = 0, timeLoop = 7200, carDataChoose = 1):
        self.timeLoop = timeLoop
        self.mymap = mapAgent(carDataChoose, self.timeLoop)
        if modelChoose == 0:
            self.modelAgent = QLearningAgent(epsilon=0.2, alpha=0.1)
            # self.modelAgent = QLearningAgent(epsilon=0.05)
        elif modelChoose == 2:
            self.modelAgent = ApproximateQAgent(epsilon=0.2, alpha=0.1)
        elif modelChoose == 1:
            self.modelAgent = SARSAAgent(epsilon=0.2, alpha=0.1)
        elif modelChoose == 3:
            self.modelAgent = DeepQAgent(epsilon=0.2, alpha=0.1)
            
        self.mapDisplay = mapDisplay(self.mymap)
        self.testTime = 100
       
    def start(self):
        time = 0
        while not self.mymap.isTerminal() or time < self.timeLoop:
            print("-----------------time" + str(time))
            if time > self.timeLoop + 999:
                break
            self.mapDisplay.show()
            state = self.mymap.getState()
            print(state)
            reward = self.mymap.getReward()
            action = self.modelAgent.getAction(state)
            print(action)
            self.mymap.update(time, action)
            nextState = self.mymap.getState()
            self.modelAgent.update(state, action, nextState, reward)
            
            time += 1
            print("performance---------------->")
            self.mymap.showTestPerformance()
            if self.mymap.isTrainingFinish():
                break
            
        print("last time ---->" + str(time-self.timeLoop))
        print("training finish ======> time: " + str(time))
          
        time = 0
        self.modelAgent.setEpsilon(0)
        self.mymap.testinit()
        self.modelAgent.testinit()
        while time < self.testTime:
            print("test-----------------time" + str(time))
            self.mapDisplay.show()
            state = self.mymap.getState()
            print(state)
            reward = self.mymap.getReward()
            action = self.modelAgent.getAction(state)
            print(action)
            self.mymap.update(time, action)
            nextState = self.mymap.getState()
            self.modelAgent.update(state, action, nextState, reward)
            self.mymap.showTestPerformance()
            time += 1
        self.modelAgent.final()
        self.mymap.final()
        
    