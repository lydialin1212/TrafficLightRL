from dataParser import carData
from util import Queue

class mapAgent(object):
    def __init__(self, carDataChoose=0, timeLoop = 720000):
        self.carData = carData()
        self.dataChoose = carDataChoose
        dataN, dataE, dataS, dataW = self.carData.generate(self.dataChoose, timeLoop)
        self.north = roadAgent(dataN, lightCondition = 1)
        self.south = roadAgent(dataS, lightCondition = 1)
        self.east = roadAgent(dataE)
        self.west = roadAgent(dataW)
        self.roadAgents = [self.north, self.east, self.south, self.west]
        self.time = 0
        
        self.performance= []
        self.last100 = [0, 0]
        
        self.totalWaitTime = 0
        self.totalLength = 0
    
    def testinit(self):
        self.performance= []
        self.last100 = [0, 0]
        dataN, dataE, dataS, dataW = self.carData.generate(self.dataChoose, 10000)
        self.north = roadAgent(dataN, lightCondition = 1)
        self.south = roadAgent(dataS, lightCondition = 1)
        self.east = roadAgent(dataE)
        self.west = roadAgent(dataW)
        self.roadAgents = [self.north, self.east, self.south, self.west]
        
        self.totalWaitTime = 0
        self.totalLength = 0
    
        
    def update(self, time, action = "nochange" ):
        "Update map status. Read coming car data from map_computer, \
         update the car and people number. change the light color"
        self.time = time
        if action == "change":
            self.changeTrafficLightCondition()
        for a in self.roadAgents:
            a.update(self.time)
            
        #reccord performance
        print("record performance------>")
        if time % 500 == 0:
            print("record performance------>")  
            waittime100 = self.getTotalWaitTime() - self.last100[0]
            length100 = self.getTotalCarLength() - self.last100[1]
            self.performance = self.performance + [[waittime100, length100]]
            self.last100 = [self.getTotalWaitTime(), self.getTotalCarLength()]
    
    def getReward(self):
        reward = 5*self.getPassCar() - self.getWaitTime() - self.getCarLength()
        return reward
    
    def getState(self):
   # state: ([NS carlength, EW carlength], [NS waittime, EW waittime])
        carlength = [self.north.getCarLength() + self.south.getCarLength(), self.east.getCarLength() + self.west.getCarLength()]
        waittime = [self.north.getWaitTime() + self.south.getWaitTime(), self.east.getWaitTime() + self.west.getWaitTime()]
        lightcondition = [self.north.getTrafficLightCondition(), self.east.getTrafficLightCondition()]
        return tuple([tuple(carlength), tuple(waittime), tuple(lightcondition)])
    
    def getWaitTime(self):
        time = 0
        for a in self.roadAgents:
            time += a.waitTime
        return time
        
    def getCarLength(self):
        res= 0 
        for a in self.roadAgents:
            res += a.getCarLength()
        return res
        
    def getPassCar(self):
        res= 0 
        for a in self.roadAgents:
            res += a.getPassCar()
        return res

    def getTotalCarLength(self):
        res = 0
        for a in self.roadAgents:
            res += a.getTotalCarLength()
        return res
    
    def getTotalWaitTime(self):
        res = 0
        for a in self.roadAgents:
            res += a.getTotalWaitTime()
        return res
        
    def getRoadAgents(self):
        return self.roadAgents
    
    def changeTrafficLightCondition(self):
        for agent in self.roadAgents:
            agent.changeTrafficLightCondition()
            
    def isTerminal(self):
        for a in self.roadAgents:
            if a.getCarLength() != 0:
                return False
        return True
        
    def isTrainingFinish(self):
        if(len(self.performance) < 10):
            return False
        print("==========================")
        print(self.performance)
        flag = 0
        for i in range(5):
            waitdif =  self.performance[len(self.performance) - 1 - i][0] - self.performance[len(self.performance) - 2 - i][0]
            lengthdif =  self.performance[len(self.performance) - 1 - i][0] - self.performance[len(self.performance) - 2 - i][0]
            if waitdif + lengthdif > 20 or waitdif + lengthdif < -20:
                return False
        print(self.time)
        print("Training finish: Time --> " + str(self.time))
        return True
        
    def showTestPerformance(self):
        
        print("==========================")
        print(self.performance)
        
    def final(self):
        print("self.totalLength")
        print(self.getTotalCarLength())
        print("self.totalWaitTime")
        print(self.getTotalWaitTime())




class roadAgent(object):
    def __init__(self, carData = {}, lightCondition = -1):
        self.carLength = 0
        self.totalCarLength = 0
        self.carQueue = Queue()
        self.trafficLightCondition = lightCondition # -1:red, 1:green
        self.waitTime = 0
        self.totalWaitTime = 0
        self.carData = carData
        self.greenTime = 0
        self.passCar = 0
    
    def setCarData(self, data):
        self.carData = data

    def changeTrafficLightCondition(self):
        self.trafficLightCondition = self.trafficLightCondition * -1
        if self.trafficLightCondition == 1:
            self.greenTime = -1

    def update(self, time):
        self.passCar = 0
        if self.carData[time] > 0:
            for i in range(self.carData[time]):
                self.carQueue.push(0)
        if self.trafficLightCondition == 1:
            self.greenTime += 1
            print("greenTime:" + str(self.greenTime))
            if self.greenTime % 2 == 0  and self.carLength > 0:
                self.carLength -= 1
                self.passCar = 1
                self.carQueue.pop()
        self.carQueue.addTime()
        self.waitTime = self.carQueue.getSum()
        self.carLength += self.carData[time]
        self.totalCarLength += self.carLength
        self.totalWaitTime += self.waitTime
            
    def getCarLength(self):
        return self.carLength
    
    def getWaitTime(self):
        return self.waitTime
        
    def getPassCar(self):
        return self.passCar
        
    def getTotalCarLength(self):
        return self.totalCarLength
        
    def getTotalWaitTime(self):
        return self.totalWaitTime
        
    def getTrafficLightCondition(self):
        return self.trafficLightCondition
    
        
        