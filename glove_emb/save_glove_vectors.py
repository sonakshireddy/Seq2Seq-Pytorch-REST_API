# This code transforms the original glove file into word2idx , id2word and vectors files that will be used later
import pickle
import numpy as np
from utils.constants import glove_model_path,glove_path
words = []
idx = 0
word2idx = {}


# Downloaded glove vectors from https://archive.org/download/glove.6B.50d-300d/glove.6B.50d.txt

vectors = []
glove ={}
with open('%s/glove.6B.50d.txt'%glove_model_path, 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        vect = np.array(line[1:]).astype(np.float)
        glove[word] = vect

with open(glove_path,'wb') as f:
    pickle.dump(glove,f)
