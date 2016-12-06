#!/usr/bin/env 

import numpy as np

from util.data_reader_a import KDD
from tqdm import tqdm

class IRT(object):
  def __init__(self, dataset, testset):
    self.dataset = dataset
    self.testset = testset
    self.init_params()

  def init_params(self):
    self.LR = 0.001
    self.kcs = {v: i for i, v in enumerate(self.dataset.kcs)}
    for v in self.testset.kcs:
      if not v in self.kcs:
        self.kcs[v] = len(self.kcs)
    self.num_kcs = len(self.kcs)
    self.students = {v: i for i, v in enumerate(self.dataset.students)}
    for v in self.testset.students:
      if not v in self.students:
        self.students[v] = len(self.students)
    self.num_stu = len(self.students)
    self.problems = {v: i for i, v in enumerate(self.dataset.problems)}
    for v in self.testset.problems:
      if not v in self.problems:
        self.problems[v] = len(self.problems)
    self.num_pro = len(self.problems)

    self.lr_s = np.random.random((self.num_stu, 2))
    self.fr_s = np.random.random((self.num_stu, 2))
    self.lr_c = np.random.random((self.num_kcs, 2))
    self.fr_c = np.random.random((self.num_kcs, 2))
    self.df_q = np.random.random((self.num_pro, ))
    self.e_zi = np.zeros((self.num_stu, self.num_kcs,))

    self.EDGE_POT = np.array([[1, 1, 0, 0],
                              [0, 0, 1, 1],
                              [1, 1, 0, 0],
                              [0, 0, 1, 1]])

  def sum_product(self, c):
    message = np.ones((self.num_stu, 4))
    forward_msg = []
    lr_c, fr_c = self.lr_c[self.kcs[c]], self.fr_c[self.kcs[c]]
    counter = 0
    for item in tqdm(self.dataset, desc='Forward'):
      if not c in item[KDD.KC_SUBSKILLS]:
        continue
      counter += 1
      q, s = self.problems[item[KDD.PROBLEM_STEP_ID]], self.students[item[KDD.ANON_STUDENT_ID]]
      msg_in, yi, df_q = message[s].reshape((-1,1)), int(item[KDD.CORRECT]), self.df_q[q]
      forward_msg.append(msg_in)
      pyz1 = self._sigmoid(-(2.0*yi-1.0)*(1.0+df_q))
      pyz0 = self._sigmoid(-(2.0*yi-1.0)*(df_q))
      plearn = self.lr_s[s][1-yi]*lr_c[1-yi]
      pforgt = self.fr_s[s][1-yi]*fr_c[1-yi]
      psy = np.array([pyz1*(1-pforgt), pyz1*pforgt, pyz0*plearn, pyz0*(1-plearn)]).reshape((-1,1))
      msg = np.sum((psy*msg_in).repeat(4, axis=1)*self.EDGE_POT, axis=0)
      message[s,:] = msg / np.sum(msg)
    message = np.ones((self.num_stu, 4))
    ez = []
    num_total = counter
    for index in tqdm(range(len(self.dataset)), desc='Backward'):
      item = self.dataset[index]
      if not c in item[KDD.KC_SUBSKILLS]:
        continue
      counter -= 1
      q, s = self.problems[item[KDD.PROBLEM_STEP_ID]], self.students[item[KDD.ANON_STUDENT_ID]]
      f_msg = forward_msg[counter]
      msg_in, yi, df_q = message[s].reshape((-1,1)), int(item[KDD.CORRECT]), self.df_q[q]
      pyz1 = self._sigmoid(-(2.0*yi-1.0)*(1.0+df_q))
      pyz0 = self._sigmoid(-(2.0*yi-1.0)*(df_q))
      plearn = self.lr_s[s][1-yi]*lr_c[1-yi]
      pforgt = self.fr_s[s][1-yi]*fr_c[1-yi]
      psy = np.array([pyz1*(1-pforgt), pyz1*pforgt, pyz0*plearn, pyz0*(1-plearn)]).reshape((-1,1))
      e_cur = psy * msg_in * f_msg
      ez.append(e_cur/np.sum(e_cur))
      msg = np.sum((psy*msg_in).repeat(4, axis=1)*self.EDGE_POT, axis=0)
      message[s,:] = msg / np.sum(msg)
    ez = list(reversed(ez))
    def sgd():
      gs, gc, gq = np.zeros(self.lr_s.shape), np.zeros(self.lr_c.shape), np.zeros(self.df_q.shape)
      gfs, gfc = np.zeros(self.fr_s.shape), np.zeros(self.fr_c.shape)
      counter = 0
      for item in tqdm(self.dataset, desc='SGD'):
        if not c in item[KDD.KC_SUBSKILLS]:
          continue
        ec = ez[counter]
        e_zi, prev_ezi = ec[0]+ec[1], ec[0]+ec[2]
        q, s = self.problems[item[KDD.PROBLEM_STEP_ID]], self.students[item[KDD.ANON_STUDENT_ID]]
        yi, df_q = int(item[KDD.CORRECT]), self.df_q[q]
        gq[q] += yi - np.exp(e_zi+self.df_q[q]) / (1.0 + np.exp(e_zi+self.df_q[q]))
        lr = self.lr_s[s][1-yi] * lr_c[1-yi]
        fr = self.fr_s[s][1-yi] * fr_c[1-yi]
        l = e_zi * prev_ezi * (1-fr) +  e_zi * (1-prev_ezi) * fr + (1-e_zi) * prev_ezi * lr + (1-e_zi) * (1-prev_ezi) * (1-lr)
        if l <= 0:
          print('zi:{}'.format(e_zi))
          print('pre zi: {}'.format(prev_ezi))
          print('lr: {}'.format(lr))
          print('l: {}'.format(l))
        dlr = ((1-e_zi) * prev_ezi - (1-e_zi) * (1-prev_ezi)) / np.log(l + 1e-5)
        dfr = (- e_zi * prev_ezi + e_zi * (1-prev_ezi)) / np.log(l + 1e-5)
        gs[s][1-yi] += dlr * lr_c[1-yi]
        gc[self.kcs[c]][1-yi] += dlr * self.lr_s[s][1-yi]
        gfs[s][1-yi] += dfr * fr_c[1-yi]
        gfc[self.kcs[c]][1-yi] += dfr * self.fr_s[s][1-yi]
        self.e_zi[s, self.kcs[c]] = prev_ezi
        counter += 1
      assert(self.lr_s.shape == gs.shape)
      assert(self.lr_c.shape == gc.shape)
      assert(self.df_q.shape == gq.shape)
      assert(self.fr_s.shape == gfs.shape)
      assert(self.fr_c.shape == gfc.shape)
      self.lr_s += self.LR * gs / float(num_total)
      self.lr_c += self.LR * gc / float(num_total)
      self.df_q += self.LR * gq / float(num_total)
      self.fr_s += self.LR * gfs / float(num_total)
      self.fr_c += self.LR * gfc / float(num_total)

    for _ in range(1):
      sgd()

  def predict(self):
    with open('results/triangel.txt', 'w') as f_out:
      f_out.write('Row\tCorrect First Time')
      for item in tqdm(self.testset, desc='Prediction'):
        q, s = self.problems[item[KDD.PROBLEM_STEP_ID]], self.students[item[KDD.ANON_STUDENT_ID]]
        c = [self.kcs[x] for x in item[KDD.KC_SUBSKILLS]]
        e_zi = np.average(self.e_zi[s, c])
        yi = 1.0 / (1.0 + np.exp(-e_zi-self.df_q[q]))
        lr = self.lr_s[s] * np.average(self.lr_c[c], axis=0)
        fr = self.fr_s[s] * np.average(self.fr_c[c], axis=0)
        self.e_zi[s, c] = yi*(e_zi*(1-fr[0]) - e_zi*fr[0] + (1-e_zi)*lr[0] - (1-e_zi)*(1-lr[0]))
        self.e_zi[s, c] += (1-yi) * (e_zi*(1-fr[1]) - e_zi*fr[1] + (1-e_zi)*lr[1] - (1-e_zi)*(1-lr[1]))
        f_out.write('\n{}\t{}'.format(item[KDD.ROW], yi))

  def _sigmoid(self, x):
    return 1.0 / (1.0 + np.exp(-x))

def main():
  trainset = KDD('../kddcup_challenge/algebra_2008_2009_train.txt')
  testset = KDD('../kddcup_challenge/algebra_2008_2009_test.txt')
  irt = IRT(trainset, testset)
  for _ in range(2):
    counter = 0
    for c in irt.kcs:
      print('Concept {}'.format(counter))
      irt.sum_product(c)
      counter += 1
  irt.predict()

if __name__ == '__main__':
  main()