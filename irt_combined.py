#!/usr/bin/env 

import numpy as np
from tqdm import tqdm
from scipy.sparse import lil_matrix
from sklearn.linear_model import LogisticRegression

from util.data_reader_a import KDD
from scipy.stats import norm

TOTAL_EPOCHS = 10

def logistic(x):
  return 1.0 / (1.0 + np.exp(-x))

class IRT(object):
  def __init__(self, dataset, testset):
    self.dataset = dataset
    self.testset = testset
    self.init_params()

  def init_params(self):
    self.proposal_width, self.init_width = 0.002, 1.0
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

    self.num_mat = self.num_stu * self.num_kcs
    self.X = lil_matrix((len(self.dataset), self.num_mat+self.num_pro))
    self.Y = np.zeros((len(self.dataset), ))
    for ite, item in tqdm(enumerate(self.dataset)):
      stu_index, pro_index, kcs_index = self._indexes(item)
      self.X[ite, [i + stu_index * self.num_kcs for i in kcs_index]] = 1.0 / float(len(kcs_index))
      self.X[ite, self.num_mat + pro_index] = 1.0
      self.Y[ite] = int(item[KDD.CORRECT])

    self.XX = lil_matrix((len(self.testset), self.num_mat+self.num_pro))
    for ite, item in tqdm(enumerate(self.testset)):
      stu_index, pro_index, kcs_index = self._indexes(item)
      self.XX[ite, [i + stu_index * self.num_kcs for i in kcs_index]] = 1.0 / float(len(kcs_index))
      self.XX[ite, self.num_mat + pro_index] = 1.0

  def posterior_ratio(self, theta_cur, theta_prev):
    prod_1 = self.X.dot(theta_cur)
    prod_2 = self.X.dot(theta_prev)

    l_data = np.exp(np.sum((prod_1-prod_2)*self.Y - np.log((1.0+np.exp(prod_1)) / (1.0 + np.exp(prod_2)))))
    l_tran = np.prod(norm(self.init_theta, self.init_width).pdf(theta_cur) / norm(self.init_theta, self.init_width).pdf(theta_prev))

    return l_data # * l_tran

  def mcmc(self, init_theta):
    self.init_theta = init_theta
    theta = norm(self.init_theta, self.init_width).rvs()
    BURN_IN, NUM_PROPOSAL, INTERVAL = 20, 10, 20
    counter = 0
    for _ in tqdm(range(BURN_IN)):
      proposal = norm(theta, self.proposal_width).rvs()
      ratio = self.posterior_ratio(proposal, theta)
      if np.random.rand() < ratio:
        theta = proposal
        counter += 1
    print('Efficiency: {}'.format(float(counter) / float(BURN_IN)))

    proposals = []
    counter = 0
    for _ in tqdm(range(NUM_PROPOSAL * INTERVAL * 10)):
      if len(proposals) == NUM_PROPOSAL:
        break
      proposal = norm(theta, self.proposal_width).rvs()
      ratio = self.posterior_ratio(proposal, theta)
      if np.random.rand() < ratio:
        if counter == INTERVAL:
          proposals.append(proposal)
          counter = 0
        else:
          counter += 1
        theta = proposal
    proposals = np.vstack(proposals)
    YY = 1.0 - 1.0 / (1.0 + np.exp(self.XX.dot(proposals.transpose())))
    YY = np.average(YY, axis=1)
    with open('submission1.txt', 'w') as f_out:
      f_out.write('Row\tCorrect First Attempt')
      for idx in range(len(self.testset)):
        f_out.write('\n{}\t{}'.format(self.testset[idx][KDD.ROW], YY[idx]))
    return YY

  def optimize_scipy(self):
    print('Training...')
    model = LogisticRegression(fit_intercept=False)
    model.fit(self.X, self.Y)
    print('Evaluating...')
    prob = model.predict_proba(self.XX)
    with open('submission1.txt', 'w') as f_out:
      f_out.write('Row\tCorrect First Attempt')
      for idx in range(len(self.testset)):
        f_out.write('\n{}\t{}'.format(self.testset[idx][KDD.ROW], prob[idx][1]))
    return prob, model.coef_

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
  p_2, coeff = irt.optimize_scipy()
  p_1 = irt.mcmc(coeff.reshape((-1,)))

  pr_1 = p_1 > 0.5
  p_2 = np.array([item[1] for item in p_2])
  pr_2 = p_2 > 0.5
  print('Accuracy: {}'.format(np.sum(pr_1 == pr_2) / float(len(pr_1))))
  print('MSE: {}'.format(np.average((p_1 - p_2)**2)))
  # irt.optimize(1e-3)
  # irt.eval()

if __name__ == '__main__':
  main()