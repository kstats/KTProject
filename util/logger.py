import time
from collections import defaultdict
from timer import Timer

class Logger:
  lstMsgType = ['ERROR', 'WARN', 'INFO']
  lstMsgTypePrint = ['ERROR']


  def __init__(self, filename, debug = True):
    self.filename = filename
    self.debug = debug
    self.dictMsgType2Msg2Count = defaultdict(lambda: defaultdict(lambda: 0))

    self.timer = Timer()
    self.output = open(filename + '.log', 'w', 0)
    self.log('========= START =========')


  def log(self, strLog):
    if self.debug or ('DEBUG' not in strLog):
      self.output.write("{0}\t{1}\n".format(time.time(), strLog))

    for msgType in self.lstMsgType:
      if msgType in strLog:
        aryStrLog = strLog.split(':')
        if msgType in aryStrLog[0]:
          self.dictMsgType2Msg2Count[msgType][aryStrLog[0]] += 1

  def endLog(self, strLog = ''):
    elapseTime = self.timer.endTimer('__init__')
    self.log('========== END ==========')
    self.output.write("""=========================
ELAPSE_TIME: {}
=========================
""".format(Timer.makeStrElapseTime(elapseTime)))

    if strLog != '':
      self.output.write("""
{}
=========================
""".format(strLog))

    
    for msgType in self.lstMsgType:
      self.output.write('{}_MSG:'.format(msgType))
      if len(self.dictMsgType2Msg2Count[msgType]) > 0:
        self.output.write('\n')
        for msg,count in self.dictMsgType2Msg2Count[msgType].items():
          self.output.write('{}\tcount-{}\n'.format(msg, count))
      else:
        self.output.write(' NONE\n'.format())
      self.output.write('=========================\n')

    for msgType in self.lstMsgTypePrint:
      if len(self.dictMsgType2Msg2Count[msgType]) > 0:
        print '{}_ENCOUNTERED'.format(msgType)
        for msg,count in self.dictMsgType2Msg2Count[msgType].items():
          print '{}\tcount-{}\n'.format(msg, count)
