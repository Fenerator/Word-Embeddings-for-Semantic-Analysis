from gensim.test.utils import datapath
from gensim import utils
import gensim.models
import tempfile
import string

class MyCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __iter__(self):
        corpus_path = 'pa3_input_text.txt'
        for line in open(corpus_path):
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
#model = gensim.models.Word2Vec(sentences=sentences, vector_size=83, epochs=60, workers=1)
model = gensim.models.Word2Vec(sentences=sentences, vector_size=83, epochs=1)

vec_king = model.wv['constantine']
print('King vector ',vec_king)
model.save('model1')

'''
for index, word in enumerate(model.wv.index_to_key):
    if index == 10:
        break
    print(f"word #{index}/{len(model.wv.index_to_key)} is {word}")



# To load a saved model:
new_model = gensim.models.Word2Vec.load('model1')
vec_fa = new_model.wv['pierre']
print('FA vector ',vec_fa)
'''




