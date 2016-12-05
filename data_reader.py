#!/usr/bin/env

import csv
import numpy as np
import sys

class Dataset(list):
  def __init__(self, file_path):
    self.file_path = file_path
    self._preload()

  def _preload(self):
    pass

  def print_example(self, num=5):
    for item in np.random.choice(self, num, replace=False):
      print('')
      self._display_single_item(item)

  def _display_single_item(self, item):
    print(item)

class KDD(Dataset):
  def _preload(self):
    with open('concept_map.json', 'r') as f_in:
      self.concept_map = json.load(f_in)

    with open(self.file_path, 'r') as f_in:
      csv_reader = csv.DictReader(f_in, delimiter='\t')
      count = 0
      for row in csv_reader:
        row['KC(SubSkills)'] = '~~'.join([self.concept_map[c] if c in self.concept_map else np.random.choice(list(self.kcs)) for c in row['KC(SubSkills)'].split('~~')])
        self.append(row)
        count+=1

  def _display_single_item(self, item):
    print(item)

def main():
  dataset = KDD('../../bridge_to_algebra_2006_2007/bridge_to_algebra_2006_2007_train.txt')
  dataset.print_example()

if __name__ == '__main__':
  main()