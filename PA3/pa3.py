#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import time
import argparse
import string
import string
from collections import Counter
import numpy as np
import pandas as pd
from collections import defaultdict
from gensim.test.utils import datapath
from gensim import utils
import gensim.models
import tempfile

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=str, help="pa3_T.txt file", required=True)
    parser.add_argument("--B", type=str, help="pa3_B.txt file", required=True)
    parser.add_argument("--text", type=str, help="Input txt file", required=True)
    args = parser.parse_args()
    return args

# Code from PA1____________________________________________________________________________
def preprocessing(textfile, contains_labels=False):
    if contains_labels:
        # remove labels and ad to list:
        # File names: to read in from and read out to
        output_file = textfile + '2'

        text_strings = []  # empty array to store the last column text
        with open(textfile) as ff:
            ss = ff.readlines()  # read all strings in a string array

        for s in ss:
            text_strings.append(s.split('\t')[0])  # add column to the text array

        with open(output_file, 'w') as outf:
            outf.write('\n'.join(text_strings))  # write everything to output file

        textfile = output_file


    with open(textfile, 'r', encoding='utf8') as infile:
        preprocessed_text = []
        lines = infile.readlines()
        for line in lines:
            line = line.split()
            line = [word.lower() for word in line] #set to lowercase
            # remove punctuation from each word
            table = str.maketrans('', '', string.punctuation)
            line = [word.translate(table) for word in line]
            #remove empty elements
            line = [word for word in line if word]
            preprocessed_text.append(line)

    #convert preprocessed_text into a single list
    flat_list = []
    for sublist in preprocessed_text:
        for item in sublist:
            flat_list.append(item)
    return flat_list

def get_aggregated_window_text(text_list, center_word, window_size):
    border = window_size//2 # assuming uneven window_size
    #get index positions of center word
    center_index = []
    for i, j in enumerate(text_list):
        if j == center_word:
            center_index.append(i)
    #get window text for a center word
    window_text = []
    for el in center_index: # at each occurrence of center_word:
        if el-border<0:
            window_text.extend(text_list[0:el])  # if lower boundary is negative
        else:
            window_text.extend(text_list[el-border:el]) # add left window text

        window_text.extend(text_list[el+1:el+border+1]) # add right window text, always

    return window_text


def count_cooccurences(text_list, center_word, window_size, context_words_list):
    counts = [0] * len(context_words_list)  # stores values for each context word
    window_text = get_aggregated_window_text(text_list, center_word, window_size)
    # keep only words in context_words_list:
    cleaned_window_text = [x for x in window_text if x in context_words_list]
    counts = Counter(cleaned_window_text)
    #write counts into embedding vector (at correct position), having dimension of context words
    count_vector = []
    for el in context_words_list:
        count_vector.append(counts[el])
    return count_vector



def get_cooccurrence_matrix(text_list, center_words_list, context_words_list, window_size):
    cooccurrence_matrix = {key: [0] * len(context_words_list) for key in center_words_list}  # structure to store values (for each center word -> 0 vector of length context_word_list)
    for center in cooccurrence_matrix: #for each center word
        cooccurrence_matrix[center] = count_cooccurences(text_list, center, window_size, context_words_list)

    return cooccurrence_matrix

def to_df(dict, context_words_list):
    df = pd.DataFrame.from_dict(dict, orient='index').transpose()
    df.index = context_words_list
    return df

def get_PPMI_values(text_list, Co_occurence_df, center_words_list, context_words_list):
    PPMI_results = {key: [0] * len(context_words_list) for key in center_words_list} #dict structure to store results
    #Get total counts for calculation of var:
    center_sums = get_sums(Co_occurence_df, 0) # axis = 0 -> center sum
    context_sums = get_sums(Co_occurence_df, 1)  # axis = 1 -> context sum
    text_len = len(text_list)
    #print(len(center_words_list), len(context_words_list))
    for cent_ind in range(len(center_words_list)): # for each center word
        for cont_index in range(len(context_words_list)):  # for each context word
            var = (center_sums[cent_ind]/text_len) * (context_sums[cont_index]/text_len)
            if var == 0:
                PPMI_results[center_words_list[cent_ind]][cont_index] = 0
            else:
                with np.errstate(divide='ignore'): # suppress error when log is 0
                    PPMI_results[center_words_list[cent_ind]][cont_index] = np.maximum(np.log2((Co_occurence_df.iat[cont_index, cent_ind] / len(text_list)) / var), 0)
    return PPMI_results

def get_sums(Co_occurence_df, axis):
    """Replaces func count_occurrence"""
    # axis = 0 -> center sum
    # axis = 1 -> context sum
    if axis == 0 or axis == 1:
        return Co_occurence_df.sum(axis=axis).to_list()
        return center_sums
    else:
        return None


# _________________________________________________________________________________________

def get_sparse(textfile, B, T):
    """returns PMI weighted coocurrence matrix from PA1"""

    # Step 1: Preprocessing
    text_list = preprocessing(textfile)
    center_words_list = preprocessing(T, contains_labels=True) # center words, ehem. B_list, same as Target Words
    context_words_list = preprocessing(B)  # context words, ehem. T_list

    # Step 2: raw Co-occurence matrix
    window_size = 5 # same as word to vec
    cooccurrence_matrix = get_cooccurrence_matrix(text_list, center_words_list, context_words_list, window_size)
    Co_occurence_df = to_df(cooccurrence_matrix, context_words_list)
    Co_occurence_df.to_csv('Co_occurence_df', encoding='utf-8')
    # print('Cooccurence matrix', Co_occurence_df.round(2).transpose())  # transposing seems nec. acc. to req.
    # use PPMI scores as weights
    PPMI = get_PPMI_values(text_list, Co_occurence_df, center_words_list, context_words_list)
    PPMI_df = to_df(PPMI, context_words_list).transpose()
    PPMI_df.to_csv('PPMI_df', encoding='utf-8')
    #print('Cooccurence matrix (PPMI weighted)', PPMI_df.round(2).transpose())

    return PPMI_df


def get_dense(textfile, T_list, len_B_list):
    """returns word2vec representation"""

    class MyCorpus:
        """An iterator that yields sentences (lists of str)."""

        def __iter__(self):
            #corpus_path = 'pa3_input_text.txt'
            corpus_path = textfile
            for line in open(corpus_path): # same preprocessing as in PA1
                # assume there's one document per line, tokens separated by whitespace
                line = line.split()
                line = [word.lower() for word in line]  # set to lowercase
                # remove punctuation from each word
                table = str.maketrans('', '', string.punctuation)
                line = [word.translate(table) for word in line]
                # remove empty elements
                line = [word for word in line if word]
                yield line

    sentences = MyCorpus()
    model = gensim.models.Word2Vec(sentences=sentences, window=5, vector_size=83, epochs=1, workers=1)
    #model = gensim.models.Word2Vec(sentences=sentences, window=5, vector_size=83, epochs=60, workers=1) # TODO use this here

    vectors = []
    for el in T_list:
        try:
            vectors.append(model.wv[el].tolist())
        except KeyError: # handle missing cases by creating 0 vector
            print('Key Error with', el, ' using 0-Vector instead')
            vectors.append(len_B_list* [0])

    # Make dataframe
    #print('Vectors: ', vectors, type(vectors[0]))
    dense_df = pd.DataFrame(list(zip(T_list, vectors)), columns=['T_list', 'vectors'])
    dense_df.to_csv('dense_df', encoding='utf-8')
    return dense_df


# CODE FROM PA2_________________________________________________________________________________________________________
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

def get_labels(T):
    # get labels:
    # Data: creating input format from txt file, use bias input: add 1 in last coordinate of points, and add an additional weight
    df = pd.read_csv(T, sep='\t', header=None, names=['T_Word', 'Label'])
    # create label vector
    text_labels = df['Label'].tolist()
    # encode text_labels 0 for war, 1 for peace
    labels = []
    for el in text_labels:
        if el == 'WAR':
            labels.append(0)
        elif el == 'PEACE':
            labels.append(1)
        else:
            raise KeyError

    return labels

def get_trainset(labels, matrix_df, get_from_row=True):
    if get_from_row: # get row vectors containing coordinates:
        # add bias (1 for each last coordinate) of points
        bias = [1] * len(labels)
        matrix_df['bias'] = bias
        # create vector from row of df including bias term
        points = matrix_df.values.tolist()
        #print(points)
    else:
        points = matrix_df['vectors'].tolist()
        # add 1 at each last coordinate
        for point in points:
            point.append(1)
        #print('Dense Matrix list: ', matrix_df['vectors'].tolist())

    # create train structure:
    training_set = list(zip(points, labels))  # create datastructure [([point coordinates], label), ...]



    return training_set


def train(training_set):
    learning_rate = 0.2
    alpha = 2
    weights = [0.0] * len(training_set[0][0])  # initialize weight vector with 0

    for iteration in range(100): # use as stopping criterion
        for point, label in training_set:
            dot_product = np.dot(point, weights)
            result_sigmoid = sigmoid(alpha, dot_product)
            error = label - result_sigmoid
            prediction = predict(point, weights)  # prediction of classifier
            #if iteration == 99:
                #print(f'Iteration: {iteration} True result: {label} \t Output: {result_sigmoid} \t Evaluation: {prediction==label}')
            if abs(error) > 0.001:
                # Weight update
                for i, val in enumerate(point):
                    weights[i] += val * error * learning_rate

    return weights

def predict(point, weights):
    '''
    retruns dot product of all but last column + identity of last weight
    :param point: list cont. coordinates
    :param weights: list of weight values
    :return: prediction (0 or 1)
    '''
    return unit_step(np.dot(point[:-1], weights[:-1]) + weights[-1])

# ______________________________________________________________________________________________________________________
def get_accuracy(training_set, weights):
    error_count = 0
    for point, label in training_set:
        prediction = predict(point, weights)  # prediction of classifier
        if label !=prediction:
            error_count+=1 # count nr. of incorrectly predicted points
    total = len(training_set)

    correct = total-error_count
    #print(f'correct/total: {correct} , {total}')
    return correct/total

def single_evaluation(T, matrix, get_from_row=True):
    labels = get_labels(T)
    training_set = get_trainset(labels, matrix, get_from_row)
    weights = train(training_set)
    accuracy = get_accuracy(training_set, weights)

    return accuracy

def average(list):
    return sum(list) / len(list)

def cross_validation(T, matrix, get_from_row=True):
    labels = get_labels(T)
    data = get_trainset(labels, matrix, get_from_row) # whole dataset
    len_data = len(data)
    bin_size = len_data // 5
    accuracies = [] # acc. per bin or fold
    lengths = []
    for i in range(0, len_data, bin_size):
        train_data = data[:]  # copy all data into trainset
        test_data = train_data[i:i+bin_size] # extract test data
        del train_data[i:i+bin_size] # remove test data from train data
        weights = train(train_data)
        accuracy = get_accuracy(test_data, weights)
        lengths.append(len(test_data))
        accuracies.append(accuracy)

    accuracies_mean = average(accuracies)
    accuracies.append(accuracies_mean) # last entry is mean
    return accuracies





def main():
    # get arguments:
    args = parse_args()
    T = args.T
    B = args.B
    textfile = args.text
    '''
    T = 'pa3_T.txt'
    B = 'pa3_B.txt'
    textfile = 'pa3_input_text.txt'
    '''

    # Get matrices
    sparse_matrix = get_sparse(textfile, B, T)
    B_list = preprocessing(B, contains_labels=False)
    len_B_list = len(B_list)
    T_list = preprocessing(T, contains_labels=True) # sets which vectors need to be considered in dense matrix
    dense_matrix = get_dense(textfile, T_list, len_B_list)

    # classify sparse
    accuracy_sparse = single_evaluation(T, sparse_matrix, get_from_row=True)
    cross_val_sparse = cross_validation(T, sparse_matrix, get_from_row=True)

    # classify dense
    accuracy_dense = single_evaluation(T, dense_matrix, get_from_row=False)
    cross_val_dense = cross_validation(T, dense_matrix, get_from_row=False)

    # make results.txt file
    # make lists
    results_sparse = cross_val_sparse[:]
    results_sparse.insert(0, accuracy_sparse)
    results_dense = cross_val_dense[:]
    results_dense.insert(0, accuracy_dense)
    description = ['single', 'cval_1', 'cval_2', 'cval_3', 'cval_4', 'cval_5', 'cval_AVG']

    results_df = pd.DataFrame(list(zip(description, results_sparse, results_dense)), columns=['evaluation', 'Results sparse', 'Results dense'])
    results_df.round(2).to_csv('results_df.txt', encoding='utf-8', sep ='\t', index=False)
    print(results_df)
    print('Saved results to results_df.txt')
if __name__ == "__main__":
    main()
