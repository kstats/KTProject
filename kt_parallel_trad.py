import numpy as np
import data_reader as dr
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm

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
student_positions = []

count = 0
for item in tqdm(kdd):
    if item['Anon Student Id'] not in students:
        students[item['Anon Student Id']] = 1
        student_positions.append(count)
    else:
        pos = list(skills.keys()).index(item['Anon Student Id'])
        if len(student_positions) > pos+1:
            print "nope nope nope"
    if item['KC(SubSkills)'] not in skills:
        skills[item['KC(SubSkills)']] = 1
    count += 1


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
        concept = list(skills.keys()).index(item['KC(SubSkills)'])
        student = list(students.keys()).index(item['Anon Student Id'])
        alpha = up_message(y, concept)
        beta = beta_message[student][concept]
        intermediate = np.multiply(alpha, beta)
        intermediate = np.array([intermediate,]*2)
        edge_pot = phi(student, concept)
        beta_message[student][concept] = np.sum(intermediate * edge_pot, axis = 1) / np.sum(beta_message[student][concept])
        running_betas[count] = beta_message[student][concept]
        count += 1

delta_message = np.ones((num_students, num_skills, 2))
E_stored = np.zeros((num_students, num_skills, 2))
def backward_pass():
    count = N-1
    kdd.reverse()
    for item in tqdm(kdd):
        y = item['Correct First Attempt']
        concept = list(skills.keys()).index(item['KC(SubSkills)'])
        student = list(students.keys()).index(item['Anon Student Id'])
        alpha = up_message(y, concept)
        delta = delta_message[student][concept]
        intermediate = np.multiply(alpha, delta)
        intermediate = np.array([intermediate, ] * 2)
        edge_pot = phi(student, concept)
        delta_message[student][concept] = np.sum(intermediate * edge_pot, axis=1) / np.sum(delta_message[student][concept])
        #this does the E-step
        E_zi = up_message(y, concept) * running_betas[count - 1] * delta_message[student][concept]
        E_zi = E_zi[0] / (E_zi[0] + E_zi[1])
        if E_stored[student][concept][0] == 0:
            E_stored[student][concept] = E_zi
        m_step(E_zi, y, student, concept)
        count -=1

model_learning_rate = 1e-3
def m_step(E_zi, y, student, concept):
    if y == 0:
        deriv_s = E_zi / (E_zi * prob_slip[concept] + (1-E_zi)(1-prob_guess[concept]))
        deriv_g = (1 + E_zi) / (E_zi * prob_slip[concept] + (1-E_zi)(1-prob_guess[concept]))
    else:
        deriv_s = E_zi / (E_zi * (1-prob_slip[concept]) + (1-E_zi) * prob_guess[concept])
        deriv_g = (1-E_zi) / (E_zi * (1-prob_slip[concept]) + (1-E_zi)*prob_guess[concept])
    deriv_lr = (1-E_zi) / (E_zi + (1-E_zi) * prob_learn[student][concept])

    prob_slip[concept] = prob_slip[concept] - model_learning_rate * deriv_s
    prob_guess[concept] = prob_guess[concept] - model_learning_rate * deriv_g
    prob_learn[student][concept] = prob_learn[student][concept] - model_learning_rate * deriv_lr


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
    if item['KC(SubSkills)'] is not None and item['KC(SubSkills)'] in skills and item['Anon Student Id'] in students:
        student = list(students.keys()).index(item['Anon Student Id'])
        concept = list(skills.keys()).index(item['KC(SubSkills)'])
        f.write(str(E_stored[student][concept][0]))
        f.write("\n")
    else:
        f.write(str(0.9))
        f.write("\n")