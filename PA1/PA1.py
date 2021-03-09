#!/usr/bin/python
import sys
import string
from collections import Counter

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
    print(flat_list)
    return flat_list

def get_aggregated_window_text(text_list, center_word, window_size):
    border = window_size//2 # assuming uneven window_size
    center_word = center_word.lower()

    #get index positions of center word
    center_index = []
    for i, j in enumerate(text_list):
        if j == center_word:
            center_index.append(i)
    print('center index', center_index)

    #get all text for center word
    center_text = []
    for el in center_index: # at each occurrence of center_word:
        center_text.extend(text_list[el-border:el]) #lower boundary
        center_text.extend(text_list[el+1:el+border+1]) # upper boundary
    return center_text

def count_cooccurences(text_list, center_word, window_size, T_list):
    counts = [0] * len(T_list)  # stores values
    center_text = get_aggregated_window_text(text_list, center_word.lower(), window_size)
    print('center text', center_text)
    #delete all words not in T_list
    cleaned_center_text = [x for x in center_text if x in T_list]
    print('cleaned center text ', cleaned_center_text)
    counts = Counter(cleaned_center_text)
    print('Counts: ', counts)

    #write counts into vector (at correct position)
    count_vector = []
    for el in T_list:
        count_vector.append(counts[el])
    print('Count Vector: ', count_vector)

    return count_vector

def count_occurrence(text_list, word): # P(list) for PPMI formula
    # delete all words not in word_list
    cleaned_text = [x for x in text_list if x in word]
    print('cleaned text ', cleaned_text)

    result = len(cleaned_text)


    return result


def get_cooccurrence_matrix(text_list, B_list, T_list):
    window_size = 5

    cooccurrence_matrix = {key: [0] * len(T_list) for key in B_list}  # structure to store values (for each T -> 0 vector)
    for basis in cooccurrence_matrix: #for each center word
        cooccurrence_matrix[basis] = count_cooccurences(text_list, basis, window_size, T_list)
    print('CO Occurence matrix ', cooccurrence_matrix)
    return cooccurrence_matrix

def get_PPMI_values(text_list, cooccurrence_matrix, T_list):
    PPMI_results = {}
    counter = 0
    for basis in cooccurrence_matrix:
        print(cooccurrence_matrix[basis])
        for t in cooccurrence_matrix[basis]:
            print('coocurence value: ', t)
            print('c: ', counter)
            print(count_occurrence(text_list, T_list[counter]))
        counter += 1



    return PPMI_results



def main(arguments):
    #Read arguments
    textfile = arguments[0]
    B = arguments[1]
    T = arguments[2]

    #Step 1: Preprocessing
    text_list = preprocessing(textfile)
    T_list = preprocessing(T)
    B_list = preprocessing(B)
    print(len(text_list))

    #Step 2: raw Co-occurence matrix
    cooccurrence_matrix = get_cooccurrence_matrix(text_list, B_list, T_list)
    print(cooccurrence_matrix)

    # use PPMI scores as weights

    #print(count_occurrence(text_list, T_list))
    print(T_list)
    print(get_PPMI_values(text_list, cooccurrence_matrix,T_list))














if __name__ == "__main__":
    if len(sys.argv) ==1:
       main(['text.txt', 'B.txt', 'T.txt'])
    else:
        main(sys.argv[1:])