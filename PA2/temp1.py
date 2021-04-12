#perceptron algo

import numpy as np

# Parameters
learning_rate = 0.2

# Data
# use bias inpus, add 1 in last coordinate of points, and add an additional weight
training_set_nand = [([0,0,1],1),([0,1,1],1),([1,0,1],1),([1,1,1],0)]
#training_set_nand = [([5,0,1],1),([6,3,1],1),([9,0,1],0),([-3,0,1],0)]
weights = [0.0,0.0,0.0]



def unit_step(x):
    return 1.0* (x>=0)  # returns 1 if x >=0

def decision_boundary(x):
    return 1.0* (x>=0.005)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

for i in range(10000):
    error_count=0
    for input,desired_out in training_set_nand:
        result = sigmoid(np.dot(input,weights))
        print("input:",input, "output:",result, 'Label: ', desired_out, "Correctly Classified: ", decision_boundary(result) == desired_out)
        error= desired_out-result
        print('Error ', error)
        if abs(error) > 0.005:
            error_count+=1
            for i,val in enumerate(input):
                weights[i]+= val * error * learning_rate
    # Stopping Criterion
    if error_count==0:
        print("#" * 60)
        print('Weights: ', weights)
        print("#" * 60)
        break
    print("#"*60)
    print('Weights: ', weights)
    print("#"*60)