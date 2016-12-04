#!/usr/bin/env 

import numpy as np
from tqdm import tqdm
from scipy.sparse import lil_matrix
from sklearn.linear_model import LogisticRegression

from util.data_reader_a import KDD

TOTAL_EPOCHS = 10

def logistic(x):
  return 1.0 / (1.0 + np.exp(-x))

class IRT(object):
  def __init__(self, dataset, testset):
    self.dataset = dataset
    self.testset = testset
    self.init_params()

  def init_params(self):
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

    self.student_knowledge = np.zeros((self.num_stu, self.num_kcs))
    self.problem_difficulty = np.zeros((self.num_pro))

  def optimize(self, learning_rate):
    for epoch in tqdm(range(TOTAL_EPOCHS), desc="Epoch"):
      for item in self.dataset:
        stu_index, pro_index, kcs_index = self._indexes(item)
        yi = int(item[KDD.CORRECT])

        e = np.exp(np.sum(self.student_knowledge[[stu_index,kcs_index]]) + self.problem_difficulty[pro_index])
        self.student_knowledge[[stu_index,kcs_index]] -= learning_rate * (e / (1.0 + e) - yi)
        self.problem_difficulty[pro_index] -=  learning_rate * (e / (1.0 + e) - yi)

  def eval(self):
    correct, loss = 0, 0.0
    for item in self.dataset:
      stu_index, pro_index, kcs_index = self._indexes(item)
      yi = int(item[KDD.CORRECT])

      w = np.sum(self.student_knowledge[[stu_index,kcs_index]]) + self.problem_difficulty[pro_index]
      p0 = 1.0 / (1.0 + np.exp(w))
      if p0 < 0.5:
        yi_predict = 1
      else:
        yi_predict = 0
      correct += yi_predict == yi
      loss += (-w * yi + np.log(1 + np.exp(w)))
    print('Accuracy: {}'.format(correct / float(len(self.dataset))))
    print('Loss: {}'.format(loss / float(len(self.dataset))))

  def optimize_scipy(self):
    num_mat = self.num_stu * self.num_kcs
    np.random.shuffle(self.dataset)
    X = lil_matrix((len(self.dataset), num_mat+self.num_pro))
    Y = np.zeros((len(self.dataset), ))
    for ite, item in tqdm(enumerate(self.dataset)):
      stu_index, pro_index, kcs_index = self._indexes(item)
      X[ite, [i + stu_index * self.num_kcs for i in kcs_index]] = 1.0 / float(len(kcs_index))
      X[ite, num_mat + pro_index] = 1.0
      Y[ite] = int(item[KDD.CORRECT])
    print('Training...')
    model = LogisticRegression()
    model.fit(X, Y)
    print('Evaluating...')
    XX = lil_matrix((len(self.testset), num_mat+self.num_pro))
    for ite, item in tqdm(enumerate(self.testset)):
      stu_index, pro_index, kcs_index = self._indexes(item)
      XX[ite, [i + stu_index * self.num_kcs for i in kcs_index]] = 1.0 / float(len(kcs_index))
      XX[ite, num_mat + pro_index] = 1.0
    prob = model.predict_proba(XX)
    with open('submission.txt', 'w') as f_out:
      f_out.write('Row\tCorrect First Attempt')
      for idx in range(len(self.testset)):
        f_out.write('\n{}\t{}'.format(self.testset[idx][KDD.ROW], prob[idx][1]))


  def _indexes(self, data_item):
    stu_index = self.students[data_item[KDD.ANON_STUDENT_ID]]
    pro_index = self.problems[data_item[KDD.PROBLEM_STEP_ID]]
    kcs_index = [self.kcs[k] for k in data_item[KDD.KC_SUBSKILLS]]
    return stu_index, pro_index, kcs_index

def main():
  dataset = KDD('../kddcup_challenge/algebra_2008_2009_train.txt')
  testset = KDD('../kddcup_challenge/algebra_2008_2009_test.txt')
  # dataset = KDD('../bridge_to_algebra_2006_2007/bridge_to_algebra_2006_2007_train.txt')
  # testset = KDD('../bridge_to_algebra_2006_2007/bridge_to_algebra_2006_2007_test.txt')
  dataset.print_meta_data()
  dataset.print_example()
  irt = IRT(dataset, testset)
  irt.optimize_scipy()
  # irt.optimize(1e-3)
  # irt.eval()

if __name__ == '__main__':
  main()