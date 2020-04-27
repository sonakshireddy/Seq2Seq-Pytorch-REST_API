# this file loads the glove vectors and creates a weight matrix from the vectors for all the words in the training data
import pickle
import numpy as np
import torch
import torch.nn as nn
from utils.constants import glove_path,weight_matrix_path


class GloveVectors(object):
    __glove = None

    def __init__(self):
       if GloveVectors.__glove is None:
           GloveVectors.__get_glove_model()

    @staticmethod
    def __get_glove_model():
        with open(glove_path, 'rb') as f:
            GloveVectors.__glove = pickle.load(f)

    @staticmethod
    def get_glove_instance():
        if GloveVectors.__glove is None:
            GloveVectors.__get_glove_model()
        return GloveVectors.__glove


def get_train_embeddings(word2index):
    matrix_len = len(word2index.keys())+2
    weights_matrix = np.zeros((matrix_len, 50))
    words_found = 0
    emb_dim = 50
    glove = GloveVectors.get_glove_instance()
    for word in word2index:
        i = word2index[word]
        try:
            weights_matrix[i] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim,))
    with open(weight_matrix_path, 'wb') as handle:
        pickle.dump(weights_matrix, handle)
    return torch.tensor(weights_matrix)


def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim