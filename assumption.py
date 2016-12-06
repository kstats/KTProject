#!/usr/bin/env 
import numpy as np
from util.data_reader import KDD
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main():
  dataset = KDD('../kddcup_challenge/algebra_2008_2009_train.txt')
  counter, plot_data = 0, []
  for index, item in enumerate(dataset):
    counter += 1 if item[KDD.CORRECT] == '1' else 0
    plot_data.append(float(counter) / float(index+1))
  df = pd.DataFrame(dict(t=range(len(dataset)), c=plot_data))
  fig, ax = plt.subplots()
  sns.regplot(x='t', y='c', data=df)
  plt.savefig('results/assumption.png')
  plt.close(fig)

if __name__ == '__main__':
  main()