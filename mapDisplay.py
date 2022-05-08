from mapAgent import mapAgent, roadAgent

class mapDisplay:
    def __init__(self, mapAgent):
        self.mymap = mapAgent
        
    def show(self):
        road = self.mymap.getRoadAgents()
        print("\n\n\n")
        print("         |        |")
        print("         |        |")
        print("         |        |")
        print("         |        |")
        print("         |   "+str(road[0].getCarLength())+"    |")
        print("         |   "+str(road[0].getTrafficLightCondition())+"    |")
        print("---------          ---------")
        print("                            ")
        print("       " + str(road[3].getCarLength()) +" "+str(road[3].getTrafficLightCondition())+"      " + str(road[1].getCarLength()))
        print("                            ")
        print("---------          ---------")
        
        print("         |   "+str(road[2].getCarLength())+"    |")
        print("         |        |")
        print("         |        |")
        print("         |        |")
        print("         |        |")
        print("\n\n\n")
        
        
        