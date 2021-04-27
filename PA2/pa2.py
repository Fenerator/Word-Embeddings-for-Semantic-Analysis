#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Due to use of bias input, last element in weights_pa2.txt is bias (and not actually a weight).
'''
import numpy as np
import pandas as pd

# Parameters
learning_rate = 0.2
alpha = 2

# Data: creating input format from txt file, use bias input: add 1 in last coordinate of points, and add an additional weight
df = pd.read_csv('pa2_input.txt', sep='\t')
df = df.drop(['Word', '|'], axis=1)

#create vector from labels
text_labels = df['Label'].tolist()
df = df.drop('Label', axis=1)

# encode text_labels 0 for war, 1 for peace
labels =  []
for el in text_labels:
    if el == 'WAR':
        labels.append(0)
    elif el == 'PEACE':
        labels.append(1)
    else:
        raise KeyError

#add bias (1 for each last coordinate) of points
bias = [1] * len(labels)
df['bias'] = bias

#create vector from row of df including bias term
points = df.values.tolist()

training_set = list(zip(points, labels)) #create datastructure [([point coordinates], label), ...]
weights = [0.0] * len(points[0]) # initialize weight vector with 0

def sigmoid(a, x):
    return 1 / (1 + np.exp(-a*x))

def unit_step(x):
    return 1.0* (x>0)  # returns 1 if x >0

def predict(point, weights):
    '''
    retruns dot product of all but last column + identity of last weight
    :param point: list cont. coordinates
    :param weights: list of weight values
    :return: prediction (0 or 1)
    '''
    return unit_step(np.dot(point[:-1], weights[:-1]) + weights[-1])

for iteration in range(100):
    for point, label in training_set:
        dot_product = np.dot(point, weights)
        result_sigmoid = sigmoid(alpha, dot_product)
        #prediction = predict(point, weights) # prediction of classifier
        error = label - result_sigmoid
        if iteration == 50 or iteration == 99:
            print(f'Iteration: {iteration} True result: {label} \t Output: {result_sigmoid}')
        if abs(error) > 0.0:
            # Weight update
            for i,val in enumerate(point):
                weights[i] += val * error * learning_rate


# write weights into txt file
with open('weights_pa2.txt', 'w', encoding='utf8') as f:
    for el in weights:
        f.write(str(el))
        f.write('\n')
