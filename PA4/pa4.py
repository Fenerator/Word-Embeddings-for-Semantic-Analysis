from gensim.models import FastText
import gensim
from gensim.test.utils import datapath
from gensim.utils import tokenize
from gensim import utils
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
wv = gensim.models.fasttext.load_facebook_vectors(cap_path)


def embed_sentence(sentence):
    # create sentence embeddings
    ...







