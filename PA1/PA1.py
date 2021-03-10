#!/usr/bin/python
import sys
import string
from collections import Counter
import numpy as np
import pandas as pd

def preprocessing(textfile):
    '''

    :param textfile: raw text
    :return: preprocessed text as a list with words as elements
    '''
    with open(textfile, 'r', encoding='utf8') as infile:
        preprocessed_text = []
        lines = infile.readlines()

        for line in lines:
            line = line.split()
            line = [word.lower() for word in line] #set to lowercase
            # remove punctuation from each word
            table = str.maketrans('', '', string.punctuation)
            line = [word.translate(table) for word in line]
            preprocessed_text.append(line)

    #convert preprocessed_text into a single list
    flat_list = []
    for sublist in preprocessed_text:
        for item in sublist:
            flat_list.append(item)

    return flat_list

def get_aggregated_window_text(text_list, center_word, window_size):
    border = window_size//2 # assuming uneven window_size
    center_word = center_word.lower()

    #get index positions of center word
    center_index = []
    for i, j in enumerate(text_list):
        if j == center_word:
            center_index.append(i)


    #get all text for center word
    center_text = []
    for el in center_index: # at each occurrence of center_word:
        center_text.extend(text_list[el-border:el]) #lower boundary
        center_text.extend(text_list[el+1:el+border+1]) # upper boundary
    return center_text

def count_cooccurences(text_list, center_word, window_size, T_list):
    counts = [0] * len(T_list)  # stores values
    center_text = get_aggregated_window_text(text_list, center_word.lower(), window_size)

    #delete all words not in T_list
    cleaned_center_text = [x for x in center_text if x in T_list]

    counts = Counter(cleaned_center_text)


    #write counts into vector (at correct position)
    count_vector = []
    for el in T_list:
        count_vector.append(counts[el])


    return count_vector

def count_occurrence(text_list, word): # P(list) for PPMI formula
    '''
    returns occurence/text length
    :param text_list:
    :param word:
    :return:
    '''
    # delete all words not in word_list
    cleaned_text = [x for x in text_list if x == word]
    result = len(cleaned_text)/len(text_list)

    return result


def get_cooccurrence_matrix(text_list, B_list, T_list):
    window_size = 5

    cooccurrence_matrix = {key: [0] * len(T_list) for key in B_list}  # structure to store values (for each T -> 0 vector)
    for basis in cooccurrence_matrix: #for each center word
        cooccurrence_matrix[basis] = count_cooccurences(text_list, basis, window_size, T_list)

    return cooccurrence_matrix

def get_PPMI_values(text_list, cooccurrence_matrix,B_list, T_list):
    PPMI_results = {key: [0] * len(T_list) for key in B_list}
    #print('calculation: ')
    for basis in PPMI_results:  # for each center word
        #print('Basis: ', basis)
        for t in range(len(T_list)):  # for each T word
            #print('T word: ', T_list[t])
            PPMI_results[basis][t] = np.maximum(np.log2((cooccurrence_matrix[basis][t]/len(text_list))/(count_occurrence(text_list, basis) * count_occurrence(text_list, T_list[t]))), 0)
            #print(cooccurrence_matrix[basis][t], '/ (', count_occurrence(text_list, basis), '*', count_occurrence(text_list, T_list[t]), ') = ', PPMI_results[basis][t], 'mit log und ohne max')
    return PPMI_results

def to_df(dict, T_list):
    df = pd.DataFrame.from_dict(dict, orient='index').transpose()
    df.index = T_list
    return df

def get_row_vector(df):
    d = df.T.to_dict(orient='list')
    return d

def get_cosine_similarity(list1, list2):
    #convert to np array
    v1 = np.array(list1)
    v2 = np.array(list2)

    #normalize vectors
    v1 =  v1 / np.sqrt(np.sum(v1**2))
    v2 = v2 / np.sqrt(np.sum(v2**2))

    #calculate scalar product
    cosine_sim = np.dot(v1, v2)

    return cosine_sim

def TxT(PPMI_df, B_list, T_list):
    #convert into row vectors (T), elements are basis
    data = get_row_vector(PPMI_df) #dict keys are T, values is list of element values
    print('Data:' ,data)
    matrix = {key: [0] * len(T_list) for key in T_list}
    #iterate through all cells:
    for key in matrix:
        c = 0
        for j in T_list:
            matrix[key][c] = get_cosine_similarity(data[key], data[j])

            print('c', c)
            c += 1

    return matrix # TOdo Test
'''
    for t in PPMI:  # for each center word
        # print('Basis: ', basis)
        for t in range(len(T_list)):  # for each T word
            # print('T word: ', T_list[t])
            PPMI[basis][t] = np.maximum(np.log2((cooccurrence_matrix[basis][t] / len(text_list)) / (
                        count_occurrence(text_list, basis) * count_occurrence(text_list, T_list[t]))), 0)
'''


def main(arguments):
    #Read arguments
    textfile = arguments[0]
    B = arguments[1]
    T = arguments[2]

    #Step 1: Preprocessing
    text_list = preprocessing(textfile)
    T_list = preprocessing(T)
    B_list = preprocessing(B)


    #Step 2: raw Co-occurence matrix
    cooccurrence_matrix = get_cooccurrence_matrix(text_list, B_list, T_list)
    # use PPMI scores as weights
    PPMI = get_PPMI_values(text_list, cooccurrence_matrix, B_list, T_list)
    PPMI_df = to_df(PPMI, T_list)
    print('PPMI DF', PPMI_df)

    #Step 3: cosine similarity matrix TxT
    print('Cosine Similarity Matrix TxT: ', TxT(PPMI_df, B_list, T_list))

















if __name__ == "__main__":
    if len(sys.argv) ==1:
        main(['text.txt', 'B.txt', 'T.txt'])

       #main(['text_V2.txt', 'B_V2.txt', 'T_V2.txt'])
    else:
        main(sys.argv[1:])