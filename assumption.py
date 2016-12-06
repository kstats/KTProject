#!/usr/bin/env 
import numpy as np
from util.data_reader import KDD
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict

def main():
  dataset = KDD('../kddcup_challenge/algebra_2008_2009_train.txt')
  counting_mat = defaultdict(lambda: [])
  for index, item in enumerate(dataset):
    s, lc = item[KDD.ANON_STUDENT_ID], item[KDD.KC_SUBSKILLS]
    for c in lc:
      counting_mat[(s, c)].append(item[KDD.CORRECT])
  plot_data, total_data = [], []
  for step in range(30):
    tmp_correct, tmp_total = 0, 0
    for _, v in counting_mat.items():
      if len(v) > step:
        tmp_correct += int(v[step])
        tmp_total += 1
    plot_data.append(float(tmp_correct)/float(tmp_total))
    total_data.append(tmp_total)

  df = pd.DataFrame(dict(t=np.arange(len(plot_data)), c=plot_data, total=total_data))
  fig, ax = plt.subplots()
  sns.regplot(x='t', y='c', data=df)
  plt.savefig('results/assumption_sc.png')
  plt.close(fig)

if __name__ == '__main__':
  main()