import numpy as np
import data_reader as dr
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
    # randomly chooses an item to be the concept
    item['KC(SubSkills)'] = random.choice(list_of_skills)
print len(skills)


print "done with loop"
#this is the number of unique students that are found in the dataset
num_students = len(students.keys())
print num_students

#this is the humber of unique skills found in the dataset
num_skills = len(skills.keys())
print num_skills

N = len(kdd)
print N

# probability that a student will learn a concept after answering a given question
prob_learn = np.random.rand(num_students, num_skills)

# probability that a student will answer a question correctly after guessing
prob_guess = np.random.rand(num_skills)

# probability that a student will "slip up" on their knowledge and answer incorrectly.
prob_slip = np.random.rand(num_skills)

#this returns the alpha value in alpha beta
def up_message(y, concept):
    if y == 0:
        return np.array((prob_slip[concept], (1-prob_guess[concept])))
    else:
        return np.array(((1-prob_slip[concept]), prob_guess[concept]))

#this returns the edge potential of a given student and concept
def phi(student, concept):
    result = np.ones((2,2))
    result[1][0] = 0
    result [1][0] = prob_learn[student][concept]
    result[1][1] = 1 - prob_learn[student][concept]
    return result

beta_message = np.ones((num_students, num_skills, 2))
running_betas = np.ones((N, 2))
def forward_pass():
    count = 0
    for item in tqdm(kdd):
        y = item['Correct First Attempt']
        concept = skills[item['KC(SubSkills)']]
        student = students[item['Anon Student Id']]
        alpha = up_message(y, concept)
        beta = beta_message[student][concept]
        intermediate = np.multiply(alpha, beta)
        intermediate = np.array([intermediate,]*2)
        edge_pot = phi(student, concept)
        beta_message[student][concept] = np.sum(intermediate * edge_pot, axis = 1) / np.sum(beta_message[student][concept])
        running_betas[count] = beta_message[student][concept]
        count += 1

delta_message = np.ones((num_students, num_skills, 2))
E_stored = np.ones((num_students, num_skills,)) * -1
E_end = np.zeros((num_students, num_skills,))
def backward_pass():
    count = N-1
    global prob_guess, prob_learn, prob_slip
    grad_prob_g = np.zeros(prob_guess.shape)
    grad_prob_l = np.zeros(prob_learn.shape)
    grad_prob_s = np.zeros(prob_slip.shape)
    kdd.reverse()
    for item in tqdm(kdd):
        y = item['Correct First Attempt']
        concept = skills[item['KC(SubSkills)']]
        student = students[item['Anon Student Id']]
        alpha = up_message(y, concept)
        delta = delta_message[student][concept]
        intermediate = np.multiply(alpha, delta)
        intermediate = np.array([intermediate, ] * 2)
        edge_pot = phi(student, concept)
        delta_message[student][concept] = np.sum(intermediate * edge_pot, axis=1) / np.sum(delta_message[student][concept])
        #this does the E-step
        e_zi = up_message(y, concept) * running_betas[count] * delta_message[student][concept]
        e_zi = (e_zi[0] + 1e-5) / (np.sum(e_zi) + 1e-5)
        if E_end[student, concept] == 0:
            E_end[student, concept] = e_zi
        prev_ezi = E_stored[student, concept]
        E_stored[student, concept] = e_zi
        m_step(e_zi, y, student, concept, prev_ezi, grad_prob_s, grad_prob_g, grad_prob_l)
        count -=1
    prob_learn += model_learning_rate * grad_prob_l / float(N)
    prob_learn[prob_learn > 1.0] = 1.0
    prob_learn[prob_learn < 0.0] = 0.0
    prob_guess += model_learning_rate * grad_prob_g / float(N)
    prob_guess[prob_guess > 1.0] = 1.0
    prob_guess[prob_guess < 0.0] = 0.0
    prob_slip += model_learning_rate * grad_prob_s / float(N)
    prob_slip[prob_slip > 1.0] = 1.0
    prob_slip[prob_slip < 0.0] = 0.0

model_learning_rate = 1e-3
def m_step(e_zi, y, student, concept, prev_ezi, gs, gg, gl):
    if y == 0:
        deriv_s = e_zi / (e_zi * prob_slip[concept] + (1-e_zi)(1-prob_guess[concept]))
        deriv_g = (1 + e_zi) / (e_zi * prob_slip[concept] + (1-e_zi)(1-prob_guess[concept]))
    else:
        deriv_s = e_zi / (e_zi * (1-prob_slip[concept]) + (1-e_zi) * prob_guess[concept])
        deriv_g = (1-e_zi) / (e_zi * (1-prob_slip[concept]) + (1-e_zi)*prob_guess[concept])

    if prev_ezi > 0:
        lr = prob_learn[student, concept]
        l = e_zi * prev_ezi + (1-e_zi) * prev_ezi * lr + (1-e_zi) * (1-prev_ezi) * (1-lr)
        dlr = ((1 - e_zi) * prev_ezi - (1 - e_zi) * (1 - prev_ezi)) / np.log(l + 1e-5)
    gs[concept] += model_learning_rate * deriv_s
    gg[concept] += model_learning_rate * deriv_g
    gl[student][concept] += model_learning_rate * e_zi


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
for item in kdd_test:
    list_of_skills = item['KC(SubSkills)'].split('~~')
    concept = random.choice(list_of_skills)
    student = item['Anon Student Id']
    if concept in skills and student in students:
        concept = skills[concept]
        student = students[student]
        pyi = E_end[student, concept] * (1-prob_slip[concept]) + (1-E_end[student, concept]) * prob_guess[concept]
        lr = prob_learn[student, concept]
        if E_end[student, concept] > 1 or E_end[student, concept] < 0:
            print E_end[student, concept]
            print lr
        E_end[student, concept] += lr * (1 - E_end[student, concept])
        f.write('\n{}\t{}'.format(item['Row'], pyi))
    else:
        f.write('\n{}\t{}'.format(item['Row'], 0.9))