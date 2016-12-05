#!/usr/bin/env 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.manifold import TSNE
from util.data_reader import KDD
from collections import defaultdict
from sklearn.cluster import KMeans
import json

N_CLUSTERS = 20
student_index, concept_index = {}, {}
student_concept_correctness = defaultdict(lambda: [])

def visualize(X, labels):
  model = TSNE(n_components=2, random_state=0)
  XX = model.fit_transform(X)
  df = pd.DataFrame(dict(d1=XX[:,0], d2=XX[:,1], color=labels))
  fig, ax = plt.subplots()
  # plt.scatter(XX[:,0], XX[:,1], c=labels)
  sns.lmplot('d1', 'd2', data=df, hue='color', fit_reg=False)
  plt.savefig('t-SNE on concepts')
  plt.close(fig)

def main():
  global student_index, concept_index
  global student_concept_correctness
  dataset = KDD('../kddcup_challenge/algebra_2008_2009_train.txt')
  for row in tqdm(dataset):
    s, lc = row[KDD.ANON_STUDENT_ID], row[KDD.KC_SUBSKILLS]
    if not s in student_index:
      student_index[s] = len(student_index)
    for c in lc:
      if not c in concept_index:
        concept_index[c] = len(concept_index)
    si, lci = student_index[s], [concept_index[c] for c in lc]
    for ci in lci:
      student_concept_correctness[(si, ci)].append(int(row[KDD.CORRECT]))
  corrects = np.zeros((len(student_index), len(concept_index)))
  attempts = np.zeros((len(student_index), len(concept_index))) + 1.0
  for key, val in student_concept_correctness.items():
    corrects[key] += np.sum(val)
    attempts[key] += len(val)

  accuracy = (corrects / attempts).transpose()
  model = KMeans(n_clusters=N_CLUSTERS, random_state=0)
  labels = model.fit_predict(accuracy)
  visualize(accuracy, labels)

  c2label = {c:str(labels[i]) for c, i in concept_index.items()}
  with open('concept_map.json', 'w') as f_out:
    json.dump(c2label, f_out)

if __name__ == '__main__':
  main()