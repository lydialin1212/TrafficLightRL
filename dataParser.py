import math, random

class carData(object):
    def __init__(self):
        pass
            
    def generate(self, dataType, timeLoop):
        self.dataType = dataType
        self.timeLoop = timeLoop
        if self.dataType == 0:
            data1={}
            data2={}
            for i in range(round(self.timeLoop/2)):
                if i%4==0:
                    data1[i] = 1
                else:
                    data1[i] = 0
                data2[i] = 0
            for i in range(round(self.timeLoop/2), self.timeLoop):
                if i%4==0:
                    data2[i] = 1
                else:
                    data2[i] = 0
                data1[i] = 0
            for i in range(self.timeLoop, self.timeLoop + 1000):
                data1[i] = 0
                data2[i] = 0
            return data1, data2, data1, data2
        
        if self.dataType == 1:
            data1={}
            data2={}
            for i in range(self.timeLoop):
                if i%5 == 0:
                    data1[i] = 1
                    data2[i] = 1
                else:
                    data1[i] = 0
                    data2[i] = 0
            for i in range(self.timeLoop, self.timeLoop + 1000):
                data1[i] = 0
                data2[i] = 0
            return data1, data2, data1, data2
            
        if self.dataType == 2:
            data1={}
            data2={}
            for i in range(self.timeLoop):
                if i%3 == 0:
                    data1[i] = 1
                else:
                    data1[i] = 0
                if i%4 == 1:
                    data2[i] = 1
                else:
                    data2[i] = 0
            for i in range(self.timeLoop, self.timeLoop + 1000):
                data1[i] = 0
                data2[i] = 0
            return data1, data2, data1, data2
            
            
        if self.dataType == 3:
            data1={}
            data2={}
            for i in range(self.timeLoop):
                data1[i] = round(random.random()/1.5)
                data2[i] = round(random.random()/1.5)
            for i in range(self.timeLoop, self.timeLoop + 1000):
                data1[i] = 0
                data2[i] = 0
            return data1, data2, data1, data2

            
