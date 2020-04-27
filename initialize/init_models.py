# This file initializes the model parameters/ models only once when the code starts (singleton instances)
import pickle
import torch
from utils.constants import vocab_dict_path, weight_matrix_path,hidden_size,MODEL_STATE_PATH
from modelling.model_building import EncoderRNN,AttnDecoderRNN


class VocabDict(object):
    __vocab = None

    def __init__(self):
        if VocabDict.__vocab is None:
            VocabDict.__get_vocab_dict()

    @staticmethod
    def __get_vocab_dict():
        with open(vocab_dict_path, 'rb') as f:
            VocabDict.__vocab = pickle.load(f)

    @staticmethod
    def get_vocab_instance():
        if VocabDict.__vocab is None:
            VocabDict.__get_vocab_dict()
        return VocabDict.__vocab


class WeightMatrix(object):
    __wm = None

    def __init__(self):
        if WeightMatrix.__wm is None:
            WeightMatrix.__get_matrix()

    @staticmethod
    def __get_matrix():
        with open(weight_matrix_path, 'rb') as f:
            WeightMatrix.__wm = torch.tensor(pickle.load(f))

    @staticmethod
    def get_instance():
        if WeightMatrix.__wm is None:
            WeightMatrix.__get_matrix()
        return WeightMatrix.__wm


class GetEncoderDecoderTrainedModel(object):
    __encoder = None
    __decoder = None

    def __init__(self):
        if GetEncoderDecoderTrainedModel.__encoder is None or GetEncoderDecoderTrainedModel.__decoder is None:
            GetEncoderDecoderTrainedModel.__get_model()

    @staticmethod
    def get_instance():
        if GetEncoderDecoderTrainedModel.__encoder is None or GetEncoderDecoderTrainedModel.__decoder is None:
            GetEncoderDecoderTrainedModel.__get_model()
        return GetEncoderDecoderTrainedModel.__encoder,GetEncoderDecoderTrainedModel.__decoder

    @staticmethod
    def __get_model():
        checkpoint = torch.load(MODEL_STATE_PATH)
        encoder = EncoderRNN(WeightMatrix.get_instance(), hidden_size)
        decoder = AttnDecoderRNN(hidden_size, VocabDict.get_vocab_instance().n_words, WeightMatrix.get_instance(), dropout_p=0.1)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        GetEncoderDecoderTrainedModel.__encoder = encoder
        GetEncoderDecoderTrainedModel.__decoder = decoder

