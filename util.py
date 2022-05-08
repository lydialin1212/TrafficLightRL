import random

def flipCoin(p):
    r = random.random()
    return r < p

class Queue:
    "A container with a first-in-first-out (FIFO) queuing policy."

    def __init__(self):
        self.list = []

    def push(self, item):
        "Enqueue the 'item' into the queue"
        self.list.insert(0, item)

    def pop(self):
        """
        Dequeue the earliest enqueued item still in the queue. This
        operation removes the item from the queue.
        """
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the queue is empty"
        return len(self.list) == 0
        
    def getItems(self):
        return self.list
        
    def addTime(self):
        for i in range(len(self.list)):
            self.list[i] += 1
    
    def getSum(self):
        res = 0
        for i in range(len(self.list)):
            res += self.list[i]
        return res
    
    def show(self):
        print("================")
        for i in range(len(self.list)):
            
            print(self.list[i])
        print("================")