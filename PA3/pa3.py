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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--T", type=str, help="pa3_T.txt file", required=True)
    parser.add_argument("--B", type=str, help="pa3_B.txt file", required=True)
    parser.add_argument("--text", type=str, help="Input txt file", required=True)

    args = parser.parse_args()

    return args

# Code from PA1____________________________________________________________________________
def preprocessing(textfile):
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
    print(len(center_words_list), len(context_words_list))
    for cent_ind in range(len(center_words_list)): # for each center word
        for cont_index in range(len(context_words_list)):  # for each context word
            var = (center_sums[cent_ind]/text_len) * (context_sums[cont_index]/text_len) # TODO error here
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
    # Switch Variable Names
    center_words_list = preprocessing(T) # TODO needs different preprocessing method!  # center words, ehem. B_list, same as Target Words
    context_words_list = preprocessing(B)  # context words, ehem. T_list

    # Step 2: raw Co-occurence matrix
    window_size = 5
    cooccurrence_matrix = get_cooccurrence_matrix(text_list, center_words_list, context_words_list, window_size)
    Co_occurence_df = to_df(cooccurrence_matrix, context_words_list)
    Co_occurence_df.to_csv('Co_occurence_df', encoding='utf-8')
    # print('Cooccurence matrix', Co_occurence_df.round(2).transpose())  # transposing seems nec. acc. to req.
    # use PPMI scores as weights
    PPMI = get_PPMI_values(text_list, Co_occurence_df, center_words_list, context_words_list)
    PPMI_df = to_df(PPMI, context_words_list)
    print('Cooccurence matrix (PPMI weighted)', PPMI_df.round(2).transpose())
    PPMI_df.to_csv('PPMI_df', encoding='utf-8')

    return PPMI_df


def get_dense():
    """returns word2vec representation"""
    ...

def single_evaluation():
    ...

def cross_validation_eval():
    ...

def main():
    # get arguments:
    args = parse_args()
    T = args.T
    B = args.B
    textfile = args.text

    # Get sparse matrix
    sparse_matrix = get_sparse(textfile, B, T)









if __name__ == "__main__":
    main()

