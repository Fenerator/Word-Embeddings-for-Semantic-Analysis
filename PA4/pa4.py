from gensim.models import FastText
import gensim
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import csv

# Data
gold_scores = pd.read_csv('STS.news.sr.txt', sep='\t', encoding='utf-8', header=None,
                              quoting=csv.QUOTE_NONE)[0].tolist()
sentences1 = pd.read_csv('STS.news.sr.txt', sep='\t', encoding='utf-8', header=None,
                             quoting=csv.QUOTE_NONE)[6].tolist()
sentences2 = pd.read_csv('STS.news.sr.txt', sep='\t', encoding='utf-8', header=None,
                             quoting=csv.QUOTE_NONE)[7].tolist()
# Preprocess sentences
sentences1 = [gensim.utils.simple_preprocess(el) for el in sentences1]
sentences2 = [gensim.utils.simple_preprocess(el) for el in sentences2]

# get word vectors
cap_path = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sr.300.bin.gz'
#cap_path = 'cc.sr.300.bin.gz'
wv = gensim.models.fasttext.load_facebook_vectors(cap_path)


def embed_sentence(sentence):
    # create mean sentence embeddings
    sentence_embeddings = [] # each column is a word vector
    for word in sentence:
        sentence_embeddings.append(list(wv[word]))

    # calculate mean (columnwise)
    sentence_embeddings = np.array(sentence_embeddings)
    mean_embedding = sentence_embeddings.mean(axis=0) # average embedding

    return mean_embedding


def calc_cosine_similarity(v1, v2):
    # cosine similarity of 2 numpy arrays
    # normalize vectors
    len_v1 = np.sqrt(np.sum(v1 ** 2))
    len_v2 = np.sqrt(np.sum(v2 ** 2))
    if len_v1 == 0:
        return 0
    else:
        v1 = v1 / np.sqrt(np.sum(v1 ** 2))
    if len_v2 == 0:
        return 0
    else:
        v2 = v2 / np.sqrt(np.sum(v2 ** 2))
    # calculate scalar product
    cosine_sim = np.dot(v1, v2)
    return cosine_sim


def get_similarities():
    similarities = []
    for s1, s2 in zip(sentences1, sentences2): # for each sentence pair
        # embed sentences
        e1 = embed_sentence(s1)
        e2 = embed_sentence(s2)

        # calculate similarity
        similarities.append(calc_cosine_similarity(e1, e2))

    return similarities

def pearson_corr(preds, labels):
    return pearsonr(preds, labels)[0]

similarity_list = get_similarities()
correl = pearson_corr(similarity_list, gold_scores)

print(f'Correlation between calculated and correct similarity scores: {correl}')
