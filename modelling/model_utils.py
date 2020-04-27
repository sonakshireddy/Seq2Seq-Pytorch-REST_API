# this file contains utilities for the model training and prediction
import torch
from utils.string_utils import tensorFromSentence
from utils.constants import EOS_token, SOS_token,MAX_LENGTH


def evaluate(encoder, decoder, sentence, vocab_dict, max_length=MAX_LENGTH):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        input_tensor = tensorFromSentence(vocab_dict, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)
            encoder_output = (encoder_output[:, :, :encoder.hidden_size] +
                              encoder_output[:, :, encoder.hidden_size:])
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]])
        encoder_hidden = encoder_hidden.view(1, 1, -1)
        decoder_hidden = (encoder_hidden[:, :, :encoder.hidden_size] +
                          encoder_hidden[:, :, encoder.hidden_size:])

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(vocab_dict.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]



