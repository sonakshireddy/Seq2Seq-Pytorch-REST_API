# This file trains the seq to seq model and saves the state dict to file
from __future__ import unicode_literals, print_function, division
import random
import time
import math
import torch
import torch.nn as nn
import pickle
from torch import optim
from modelling.model_building import EncoderRNN, AttnDecoderRNN
from glove_emb.get_glove_embeddings import GloveVectors, get_train_embeddings
from utils.constants import EOS_token, SOS_token, MODEL_STATE_PATH, MAX_LENGTH, hidden_size, ques_ans_data_path, \
    vocab_dict_path, device
from preprocess.parse_csv import prepare_data
from modelling.model_utils import evaluate
from utils.string_utils import tensorsFromPair

vocab_dict = None
pairs = None
weight_matrix = None

glove = GloveVectors()


def create_vocab_dict():
    global vocab_dict
    global pairs
    if vocab_dict is None or pairs is None:
        with open(ques_ans_data_path, 'rb') as handle:
            b = pickle.load(handle)
        question_desc = list(b['question'])
        answer_target = list(b['answer'])
        vocab_dict, pairs = prepare_data(question_desc, answer_target)
        with open(vocab_dict_path, 'wb') as handle:
            pickle.dump(vocab_dict, handle)
    return vocab_dict, pairs


def create_weight_matrix():
    global weight_matrix
    if weight_matrix is None:
        weight_matrix = get_train_embeddings(vocab_dict.word2index)
    return weight_matrix


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH):
    encoder.train()
    decoder.train()
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

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
    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input

        loss += criterion(decoder_output, target_tensor[di])
        if decoder_input.item() == EOS_token:
            break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def train_iters(encoder, decoder, n_iters, print_every=1000, learning_rate=0.01):
    start = time.time()
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    tr = []
    for i in range(len(pairs)):
        tr.append(tensorsFromPair(random.choice(pairs), vocab_dict))

    num_times = int(n_iters / len(tr))
    training_pairs = []
    for i in range(num_times):
        random.shuffle(tr)
        training_pairs.extend(tr)
    training_pairs.extend(tr[:n_iters - len(training_pairs)])
    # training_pairs = [tensorsFromPair(random.choice(pairs))
    #                   for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (time_since(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict()
            }, MODEL_STATE_PATH)
            evaluateRandomly(encoder, decoder)


def evaluateRandomly(encoder, decoder, n=2):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0], vocab_dict)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


if __name__ == '__main__':
    vocab_dict, pairs = create_vocab_dict()
    weight_matrix = create_weight_matrix()

    encoder1 = EncoderRNN(weight_matrix, hidden_size)
    attn_decoder1 = AttnDecoderRNN(hidden_size, vocab_dict.n_words, weight_matrix=weight_matrix, dropout_p=0.1)

    train_iters(encoder1, attn_decoder1, 750000, print_every=10000)
