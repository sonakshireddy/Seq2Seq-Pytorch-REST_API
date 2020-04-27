# This file contains utility functions required to parse sentences or words in the training/test data
import re
import unicodedata
import torch
from utils.constants import EOS_token,device


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])",r" \1 ", s)
    s= re.sub(r"[\'’]","",s)
    s = re.sub(r"[0-9\\/@#$%^&*(),\"\-_:;]+", r" ", s)
    s = re.sub("\\s+", " ",s)
    s = s.strip()
    return s


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ') if word in lang.word2index.keys()]


def tensorFromSentence(vocab_dict, sentence):
    indexes = indexesFromSentence(vocab_dict, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair,vocab_dict):
    input_tensor = tensorFromSentence(vocab_dict, pair[0])
    target_tensor = tensorFromSentence(vocab_dict, pair[1])
    return (input_tensor, target_tensor)