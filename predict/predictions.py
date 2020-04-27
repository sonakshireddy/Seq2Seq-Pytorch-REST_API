# This file predicts the empathetic response for the sentence provided
from __future__ import unicode_literals, print_function, division
from modelling.model_utils import evaluate
from utils.constants import MAX_LENGTH
from initialize.init_models import VocabDict, WeightMatrix,GetEncoderDecoderTrainedModel


vocab_dict = VocabDict.get_vocab_instance()
weight_matrix = WeightMatrix.get_instance()
encoder, decoder = GetEncoderDecoderTrainedModel.get_instance()


def predict(sentence):
    global encoder
    global decoder
    output_words, attentions = evaluate(encoder, decoder, sentence.lower(), vocab_dict, MAX_LENGTH)
    return output_words


if __name__ == '__main__':
    o = predict(
        "im having issues with my mother in law or so")
    print(o)
    print("end")
