import numpy as np
import data_reader as dr
import math
from tqdm import tqdm
import random

#This will represent a colleciton of the unique students in the dataset
#each student will have different prior knowledge parameters
students = {}

#This will represent a colleciton of the unique skills in the dataset
#each skill will have a different learn/guess/slip parameter
skills = {}
#Read in data here:

responses = []

#data = dr.Dataset("data/bridge_to_algebra_2008_2009_train.txt")
kdd = dr.KDD("data/algebra_2008_2009_train.txt")

print "gets here"

for item in tqdm(kdd):
    if item['Anon Student Id'] not in students:
        students[item['Anon Student Id']] = len(students)
    list_of_skills = item['KC(SubSkills)'].split('~~')
    for concept in list_of_skills:
        if concept not in skills:
            skills[concept] = len(skills)
    #randomly chooses an item to be the concept
    item['KC(SubSkills)'] = random.choice(list_of_skills)

#kdd = kdd[:(int(len(kdd) / 10))]
print "done with loop"
#this is the number of unique students that are found in the dataset
num_students = len(students.keys())

#this is the humber of unique skills found in the dataset
num_skills = len(skills.keys())

N = len(kdd)

# probability that a student will learn a concept after answering a given question GIVEN the student
prob_learn_s = np.random.uniform(0.01, 0.99, num_students)

# probability that a student will learn a concept after answering a given question GIVEN the concept
prob_learn_c = np.random.uniform(0.01, 0.99, num_skills)

q = np.random.rand(num_skills)

#this returns the alpha value in alpha beta
def up_message(y, concept):
    return np.array((np.exp(float(y) * (0. + q[concept])) / (1. + np.exp(0. + q[concept])), np.exp(float(y) * (1. + q[concept]) / (1. + np.exp(1. + q[concept])))))

#this returns the edge potential of a given student and concept
def phi(student, concept):
    result = np.ones((2,2))
    result[1][0] = 0
    result [1][0] = prob_learn_s[student] * prob_learn_c[concept]
    result[1][1] = 1 - prob_learn_s[student] * prob_learn_c[concept]
    return result

beta_message = np.ones((num_students, num_skills, 2))
running_betas = np.ones((N, 2))
def forward_pass():
    count = 0
    for item in tqdm(kdd):
        y = item['Correct First Attempt']
        concept = skills[item['KC(SubSkills)']]
        student = students[item['Anon Student Id']]
        running_betas[count] = beta_message[student][concept]
        alpha = up_message(y, concept)
        beta = beta_message[student][concept]
        intermediate = (alpha * beta).reshape((-1, 1)).repeat(2, axis=1)
        edge_pot = phi(student, concept)
        new_msg = np.sum(intermediate * edge_pot, axis=0)
        beta_message[student][concept] = new_msg / np.sum(new_msg)
        count += 1

delta_message = np.ones((num_students, num_skills, 2))
E_stored = np.ones((num_students, num_skills,)) * -1
E_end = np.zeros((num_students, num_skills,))
model_learning_rate = 1.0
def backward_pass():
    global prob_learn_s, prob_learn_c, q
    count = N-1
    kdd.reverse()
    grad_learn_s = np.zeros(prob_learn_s.shape)
    grad_learn_c = np.zeros(prob_learn_c.shape)
    grad_learn_q = np.zeros(q.shape)
    for item in tqdm(kdd):
        y = item['Correct First Attempt']
        concept = skills[item['KC(SubSkills)']]
        student = students[item['Anon Student Id']]
        alpha = up_message(y, concept)
        delta = delta_message[student][concept]
        intermediate = (alpha*delta).reshape((-1, 1)).repeat(2, axis=1)
        edge_pot = phi(student, concept).transpose()
        new_msg = np.sum(intermediate * edge_pot, axis=0)
        delta_message[student][concept] = new_msg / np.sum(new_msg)
        #this does the E-step
        e_zi = up_message(y, concept) * running_betas[count] * delta_message[student][concept]
        e_zi = (e_zi[0] + 1e-5) / (np.sum(e_zi) + 1e-5)
        if E_end[student, concept] == 0:
            E_end[student, concept] = e_zi
        prev_ezi = E_stored[student, concept]
        E_stored[student, concept] = e_zi
        m_step(e_zi, y, student, concept, prev_ezi, grad_learn_s, grad_learn_c, grad_learn_q)
        count -= 1
    prob_learn_s += model_learning_rate * grad_learn_s / float(N)
    print "Student learning rate"
    print np.max(grad_learn_s / float(N))
    print np.min(grad_learn_s / float(N))
    print np.average(grad_learn_s / float(N))

    prob_learn_s[prob_learn_s>1.0] = 1.0
    prob_learn_s[prob_learn_s<0.0] = 0.0
    prob_learn_c += model_learning_rate * grad_learn_c / float(N)
    prob_learn_c[prob_learn_c>1.0] = 1.0
    prob_learn_c[prob_learn_c<0.0] = 0.0
    print "Concept learning rate"
    print np.max(grad_learn_c / float(N))
    print np.min(grad_learn_c / float(N))
    print np.average(grad_learn_c / float(N))

    q += model_learning_rate * grad_learn_q / float(N)
    print "Difficulty"
    print np.max(grad_learn_q / float(N))
    print np.min(grad_learn_q / float(N))
    print np.average(grad_learn_q / float(N))



def m_step(e_zi, y, student, concept, prev_ezi, gs, gc, gq):
    gq[concept] += model_learning_rate * (float(y) - np.exp(e_zi + q[concept]) / (1 + np.exp(e_zi + q[concept])))
    if prev_ezi > 0:
        lr = prob_learn_s[student] * prob_learn_c[concept]
        l = e_zi * prev_ezi + (1-e_zi) * prev_ezi * lr + (1-e_zi) * (1-prev_ezi) * (1-lr)
        if l <= 0:
            print('zi:{}'.format(e_zi))
            print('pre zi: {}'.format(prev_ezi))
            print('lr: {}'.format(lr))
            print('l: {}'.format(l))
        dlr = ((1-e_zi) * prev_ezi - (1-e_zi) * (1-prev_ezi)) / np.log(l + 1e-5)
        gs[student] += dlr * prob_learn_c[concept]
        gc[concept] += dlr * prob_learn_s[student]

#Here is where we use alpha beta and EM to estimate these actual parameters
for i in range(3):
    print "doing forward pass now..."
    forward_pass()
    print "doing backward pass now..."
    backward_pass()

print "calculating percentages now..."
correct = 0.
total = 0.
#TODO add in final testing here once parameters are set, determine test set and test our values.
kdd_test = dr.KDD("data/algebra_2008_2009_test.txt")
f = open("algebra_2008_2009_submission.txt", 'w')
f.write('Row\tCorrect First Attempt')
for item in kdd_test:
    list_of_skills = item['KC(SubSkills)'].split('~~')
    concept = random.choice(list_of_skills)
    student = item['Anon Student Id']
    if concept in skills and student in students:
        concept = skills[concept]
        student = students[student]
        pyi = 1.0 / (1.0 + np.exp(- E_end[student, concept] - q[concept]))
        lr = prob_learn_s[student] * prob_learn_c[concept]
        if E_end[student, concept] > 1 or E_end[student, concept] < 0:
            print E_end[student, concept]
            print lr
        E_end[student, concept] += lr * (1 - E_end[student, concept])
        f.write('\n{}\t{}'.format(item['Row'], pyi))
    else:
        f.write('\n{}\t{}'.format(item['Row'], 0.9))