import datetime
import time

class Timer:
  def __init__(self):
    self.dictStartTime = {'__init__': time.time()}
    self.dictEndTime = {}

  def startTimer(self, desc):
    self.dictStartTime[desc] = time.time()

  def endTimer(self, desc):
    self.dictEndTime[desc] = time.time()
    return self.dictEndTime[desc] - self.dictStartTime[desc]

  def currElapseTime(self, desc):
    return time.time() - self.dictStartTime[desc]

  def elapseTime(self, desc):
    return self.dictEndTime[desc] - self.dictStartTime[desc]

  def strElapseTime(self, desc):
    return Timer.makeStrElapseTime(self.dictEndTime[desc]
                                   - self.dictStartTime[desc])

  @staticmethod
  def makeStrElapseTime(elapseTime):
    return datetime.timedelta(seconds=elapseTime)
