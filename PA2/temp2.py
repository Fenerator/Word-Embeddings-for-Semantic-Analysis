#creating input format from txt file
import pandas as pd
df = pd.read_csv('pa2_input.txt', sep='\t')
df = df.drop(['Word', '|'], axis=1)

#create vector from labels
text_labels = df['Label'].tolist()
df = df.drop('Label', axis=1)

# encode text_labels
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

#create vector from row of df
points = df.values.tolist()

#create datastructure [([point coordinates], label), ...]
training_set = list(zip(points, labels))

print(df)

print(training_set)

print(len(training_set))

print(len(labels))

print(len(points[0]))

def test(x):
    return x >0