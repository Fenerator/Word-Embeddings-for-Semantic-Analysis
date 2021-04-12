#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

# Parameters
learning_rate = 0.2

# Data: creating input format from txt file, use bias inpus, add 1 in last coordinate of points, and add an additional weight
df = pd.read_csv('pa2_input.txt', sep='\t')
df = df.drop(['Word', '|'], axis=1)

#create vector from labels
text_labels = df['Label'].tolist()
df = df.drop('Label', axis=1)

# encode text_labels 1 for war, 0 for peace
labels =  []
for el in text_labels:
    if el == 'WAR':
        labels.append(1)
    elif el == 'PEACE':
        labels.append(0)
    else:
        raise KeyError

#add bias (1 for each last coordinate) of points
bias = [1] * len(labels)
df['bias'] = bias

#create vector from row of df including bias term
points = df.values.tolist()

#create datastructure [([point coordinates], label), ...]
training_set = list(zip(points, labels))
weights = [0.0] * len(points[0]) # initialize weights with 0


def decision_boundary(x): # needed later to compare whether output == label
    return 1.0* (x>=0.005)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def unit_step(x):
    return 1.0* (x>=0)  # returns 1 if x >=0

for i in range(100000):
    error_count=0
    for point, label in training_set:
        dot_product = np.dot(point, weights)
        result_sigmoid = sigmoid(dot_product)
        result = unit_step(dot_product)
        #print("input:",input, "output:",result, 'Label: ', desired_out, "Correctly Classified: ", decision_boundary(result) == desired_out)
        #print("output result:", result, 'Label: ', label, "Correctly Classified: ", decision_boundary(result) == label)
        error = label - result_sigmoid
        #print('Error ', error)
        if label !=result:
            error_count+=1
            for i,val in enumerate(point):
                weights[i]+= val * error * learning_rate
    print('Nr. of errors in this iteration: ', error_count)
    # Stopping Criterion
    if error_count==0:
        print("#" * 60)
        print('Nr of iterations: ', i)
        print('Weights: ', weights)
        print("#" * 60)
        break



