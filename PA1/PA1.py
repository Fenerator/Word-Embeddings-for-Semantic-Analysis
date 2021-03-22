#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import string
from collections import Counter
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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

def get_sums(Co_occurence_df, axis):
    """Replaces func count_occurrence"""
    # axis = 0 -> center sum
    # axis = 1 -> context sum
    if axis == 0 or axis == 1:
        return Co_occurence_df.sum(axis=axis).to_list()
        return center_sums
    else:
        return None

def get_cooccurrence_matrix(text_list, center_words_list, context_words_list, window_size):
    cooccurrence_matrix = {key: [0] * len(context_words_list) for key in center_words_list}  # structure to store values (for each center word -> 0 vector of length context_word_list)
    for center in cooccurrence_matrix: #for each center word
        cooccurrence_matrix[center] = count_cooccurences(text_list, center, window_size, context_words_list)

    return cooccurrence_matrix

def get_PPMI_values(text_list, Co_occurence_df, center_words_list, context_words_list):
    PPMI_results = {key: [0] * len(context_words_list) for key in center_words_list} #dict structure to store results
    #Get total counts for calculation of var:
    center_sums = get_sums(Co_occurence_df, 0) # axis = 0 -> center sum
    context_sums = get_sums(Co_occurence_df, 1)  # axis = 1 -> context sum
    text_len = len(text_list)

    for cent_ind in range(len(center_words_list)): # for each center word
        for cont_index in range(len(context_words_list)):  # for each context word
            var = (center_sums[cent_ind]/text_len) * (context_sums[cont_index]/text_len)
            if var == 0:
                PPMI_results[center_words_list[cent_ind]][cont_index] = 0
            else:
                with np.errstate(divide='ignore'): # suppress error when log is 0
                    PPMI_results[center_words_list[cent_ind]][cont_index] = np.maximum(np.log2((Co_occurence_df.iat[cont_index, cent_ind] / len(text_list)) / var), 0)

    return PPMI_results

def to_df(dict, context_words_list):
    df = pd.DataFrame.from_dict(dict, orient='index').transpose()
    df.index = context_words_list
    return df

def get_row_vector(df):
    d = df.to_dict(orient='list')
    return d

def get_cosine_similarity(list1, list2):
    #convert to np array
    v1 = np.array(list1)
    v2 = np.array(list2)
    #normalize vectors
    len_v1 = np.sqrt(np.sum(v1**2))
    len_v2 = np.sqrt(np.sum(v2**2))
    if len_v1 == 0:
       return 0
    else:
        v1 = v1 / np.sqrt(np.sum(v1 ** 2))
    if len_v2 == 0:
        return 0
    else:
        v2 = v2 / np.sqrt(np.sum(v2 ** 2))
    #calculate scalar product
    cosine_sim = np.dot(v1, v2)
    return cosine_sim

def TxT(PPMI_df, center_words_list):
    #convert into row vectors (T), elements are basis
    # NEW convert into row vectors
    data = get_row_vector(PPMI_df) #dict keys are CENTER words, values are word embeddings [context1, context2, context3]
    matrix = {key: [0] * len(center_words_list) for key in center_words_list} #key is center word
    #iterate through all cells of matrix:
    for center in matrix:
        c = 0
        for j in center_words_list:
            matrix[center][c] = get_cosine_similarity(data[center], data[j])
            c += 1
    return matrix

def convert_sim_to_dist(cos_sim_matrix, center_words_list):
    dist_matrix = {key: [0] * len(center_words_list) for key in center_words_list}
    # iterate through all cells:
    for key in dist_matrix:
        c = 0
        for j in center_words_list:
            dist_matrix[key][c] = 1-cos_sim_matrix[key][c]
            c += 1
    return dist_matrix

#______________________________________________________________________________
def hierarchical_clusters_print(feature_matrix, target_words, max_d=0.5):
    Z_spat = linkage(feature_matrix, 'complete', 'cosine')
    clusters = fcluster(Z_spat, max_d, criterion='distance')
    num_clusters = len(set(clusters))
    # Printing clusters
    for ind in range(1, num_clusters + 1):
        print("Cluster %d words:" % ind)
        for i, w in enumerate(target_words):
            if clusters[i] == ind:
                print( ' %s' % w)
        print()

def kmeans_clusters_print(feature_matrix, target_words, num_clusters=5):
    # Fitting clusters
    km = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
    kmeans = km.fit(feature_matrix)
    cluster_labels = kmeans.labels_
    # the array of cluster labels
    # to which each input vector in n_samples belongs
    cluster_to_words = defaultdict(list)
    # which word belongs to which cluster
    for c, i in enumerate(cluster_labels):
        cluster_to_words[i].append(target_words[c])
    # Printing clusters
    for i in range(num_clusters):
        print("Cluster %d words:" % (i + 1))
        for w in cluster_to_words[i]:
            print(' %s' % w)
        print()  # add whitespace


def main(arguments):
    #Read arguments
    textfile = arguments[0]
    B = arguments[1]
    T = arguments[2]

    #Step 1: Preprocessing
    text_list = preprocessing(textfile)
    #Switch Variable Names
    center_words_list = preprocessing(T) #center words, ehem. B_list, same as Target Words
    context_words_list = preprocessing(B) #context words, ehem. T_list

    #Step 2: raw Co-occurence matrix
    window_size = 5
    cooccurrence_matrix = get_cooccurrence_matrix(text_list, center_words_list, context_words_list, window_size)
    Co_occurence_df = to_df(cooccurrence_matrix, context_words_list)
    Co_occurence_df.to_csv('Co_occurence_df', encoding='utf-8')
    #print('Cooccurence matrix', Co_occurence_df.round(2).transpose())  # transposing seems nec. acc. to req.
    # use PPMI scores as weights
    PPMI = get_PPMI_values(text_list, Co_occurence_df, center_words_list, context_words_list)
    PPMI_df = to_df(PPMI, context_words_list)
    print('Cooccurence matrix (PPMI weighted)', PPMI_df.round(2).transpose())
    PPMI_df.to_csv('PPMI_df', encoding='utf-8')

    #Step 3: cosine similarity matrix for CENTER/TARGET words, before for Context words
    cos_sim_matrix = TxT(PPMI_df, center_words_list)
    cos_sim_matrix_df = to_df(cos_sim_matrix, center_words_list)
    print('Cosine Similarity Matrix TxT: ', cos_sim_matrix_df.round(2))
    cos_sim_matrix_df.to_csv('cos_sim_matrix_df', encoding='utf-8')

    #Step 3.1: convert cosine similarity into distance matrix using cosine distance
    cos_dist_matrix = convert_sim_to_dist(cos_sim_matrix, center_words_list)
    cos_dist_matrix_df = to_df(cos_dist_matrix, center_words_list)
    print('Cosine Distance Matrix TxT: ', cos_dist_matrix_df.round(2))
    cos_dist_matrix_df.to_csv('cos_dist_matrix_df', encoding='utf-8')

    #Step 4: clustering
    feature_matrix = PPMI_df.transpose().to_numpy() #create feature_matrix
    hierarchical_clusters_print(feature_matrix, center_words_list, max_d=0.5)
    kmeans_clusters_print(feature_matrix, center_words_list, num_clusters=3)

if __name__ == "__main__":
    if len(sys.argv) ==1:
        #main(['text.txt', 'B.txt', 'T.txt'])
        main(['text_V2.txt', 'B_V2.txt', 'T_V2.txt']) # B = context words, T = center words
    else:
        main(sys.argv[1:])
