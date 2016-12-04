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
    for item in np.random.choice(len(self), num, replace=False):
      print('')
      self._display_single_item(self[item])

  def _display_single_item(self, item):
    print(item)


class KDD(Dataset):
  ANON_STUDENT_ID = 0
  PROBLEM_STEP_ID = 1
  KC_SUBSKILLS = 2
  CORRECT = 3
  ROW = 4
  def _preload(self):
    self.students, self.problems, self.kcs = set(), set(), set()
    with open(self.file_path, 'r') as f_in:
      csv_reader = csv.DictReader(f_in, delimiter='\t')
      for row in csv_reader:
        self.students.add(row['Anon Student Id'])
        self.problems.add(row['Problem Name'] + "#" + row['Step Name'])
        for x in row['KC(SubSkills)'].split('~~'):
          self.kcs.add(x)
        self.append([
          row['Anon Student Id'],
          row['Problem Name']+"#"+row['Step Name'],
          row['KC(SubSkills)'].split('~~'),
          row['Correct First Attempt'],
          row['Row']
        ])

  def _display_single_item(self, item):
    print(item)

  def print_meta_data(self):
    print('Size: {}'.format(len(self)))
    num_correct = len(list(filter(lambda x: int(x[self.CORRECT])==1, self)))
    print('Correct Answers: {}({:.2f}%)'.format(num_correct, 100 * num_correct / float(len(self))))

def main():
  dataset = KDD('../../kddcup_challenge/bridge_to_algebra_2008_2009_train.txt')
  dataset.print_example()
  dataset.print_meta_data()
  print('Number of students: {}'.format(len(dataset.students)))
  print('Number of problems: {}'.format(len(dataset.problems)))
  print('Number of skills: {}'.format(len(dataset.kcs)))

if __name__ == '__main__':
  main()