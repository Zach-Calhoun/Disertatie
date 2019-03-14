#for profiling
import time

class PerformanceTimer:
    #self.times = []
    #index = 0

    def __init__(self):
        self.times = []

    def tick(self, name : str):
        self.times += [(name,time.time())]

    def tock(self):
        now = time.time()
        try:
            name, lastTime = self.times.pop()
            deltaT = now - lastTime
            print("Tock, time took for {} : {}".format(name, deltaT))
        except:
            print("Can't do tock, likely called without calling tick() first")

