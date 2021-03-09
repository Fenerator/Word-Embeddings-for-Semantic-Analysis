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

    return flat_list



def get_aggregated_window_text(textfile, center_word, window_size):
    border = window_size//2 # assuming uneven window_size

    #get index positions of center word
    center_index = []
    for i, j in enumerate(textfile):
        if j == center_word:
            center_index.append(i)

    #get all text for center word
    center_text = []
    for el in center_index: # at each occurrence of center_word:
        center_text.append(textfile[el-border:el]) #TODO #problem wenn erster Teil segativ wird!  #lower boundary
        center_text.append(textfile[el+1:el+border+1]) # upper boundary

    return center_text










def count_occurences(textfile, center_word, window_size, T):
    counts = [0] * len(T)  # stores values
    center_text = get_aggregated_window_text(textfile, center_word, window_size)

    #delete all words not in T

    counter(center_text)
    return counts



def TxB(textfile, B, T):
    window_size = 5

    #Preprocess textfiles B and T as well
    B = preprocessing(B)
    T = preprocessing(T)
    print('T: ', T)
    print('B: ', B)

    cooccurence_matrix = {key: [0]*len(T) for key in B} #structure to store values (for each T -> 0 vector)











def main(arguments):
    textfile = arguments[0]
    B = arguments[1]
    T = arguments[2]

    #Step 1: Preprocessing
    text_list = preprocessing(textfile)
    print(len(text_list))
    #Step 2: Co-occurence matrix
    cooccurence_matrix = TxB(textfile, B, T)













if __name__ == "__main__":
    if len(sys.argv) ==1:
       main(['text.txt', 'B.txt', 'T.txt'])
    else:
        main(sys.argv[1:])