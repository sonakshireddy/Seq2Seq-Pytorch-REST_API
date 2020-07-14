# This file parses the training file containing the questions and responses to be used by the model
from __future__ import unicode_literals, print_function, division
import re
from nltk import sent_tokenize


from utils.string_utils import normalizeString
SOS_token = 0
EOS_token = 1

MAX_LENGTH = 72
class CreateDict:
    def __init__(self, name):
        self.name = name
        self.word2index = {"SOS":0, "EOS":1}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]



def get_answer_from_text(text):
    ans_t = sent_tokenize(text)
    # if re.match(".*()")
    # for a in ans_t:
    ans = ans_t[0]
    ans = re.sub("//s+", " ", ans)
    return ans

def read_data(ques, ans):
    print("Reading lines...")
    pairs=[]
    for q,a in zip(ques,ans):
        if type(q)==str and type(a) == str:
            q = normalizeString(q)
            a = normalizeString(a)
            pairs.append((q,a))
    vocab_dict= CreateDict('Vocabulary')

    return vocab_dict,pairs


def prepare_data(ques, ans):
    vocab_dict, pairs = read_data(ques, ans)
    print("Read %s sentence pairs" % len(pairs))
    input_lengths = []
    op_lengths = []
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lengths.append(len(pair[0].split(" ")))
        op_lengths.append(len(pair[1].split(" ")))
        vocab_dict.addSentence(pair[0])
        vocab_dict.addSentence(pair[1])
    print("MAX IP LENGTH {}".format(max(input_lengths)))
    print("MAX OP LENGTH {}".format(max(op_lengths)))
    print("Counted words:")
    # print(input_ques.name, input_ques.n_words)
    # print(output_ans.name, output_ans.n_words)
    return vocab_dict, pairs


