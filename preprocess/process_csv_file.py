# This file filters the questions and responses to a particular length and extracts correct responses from the answer text
from __future__ import unicode_literals, print_function, division
from io import open
import re
import pickle
import pandas as pd
from utils.string_utils import normalizeString
from utils.constants import ques, ans,CSV_PATH

MAX_LENGTH = 72


def get_answer(ques, answers_list):
    for a in answers_list:
        answers = []
        ans = normalizeString(a)
        if re.match(".*(seems like|sounds like|you seem|you sound).*", ans):
            sentences = re.split(r'[.?!]', ans)
            for sent in sentences:
                if re.match(".*(seems like|sounds like|you seem|you sound).*", sent):
                    answers.append(sent)
            if answers:
                ans = ". ".join(answers)
                return ques, ans
    return None, None


def get_answer_remaining(answers_list):
    for response in answers_list:
        a1 = re.split(r'[.?!]', response)[0]
        if 10 < len(a1.split(" ")) < MAX_LENGTH:
            return a1
    return None


if __name__ == '__main__':
    data = pd.read_csv(CSV_PATH)
    data['questionTitle'] = data['questionTitle'].apply(lambda x: normalizeString(x))
    qdict = {}
    for k, q, a in zip(data['questionTitle'], data['questionText'], data['answerText']):
        qdict.setdefault(k, {ques: "", ans: []})
        if type(q) == str and type(a) == str:
            qdict[k][ques] = q
            qdict[k][ans].append(a)

    modified_ques = []
    modified_ans = []
    qa_dict = {}
    del data
    for key in qdict:
        q, a = get_answer(qdict[key][ques], qdict[key][ans])
        if q and a:
            q1 = normalizeString(q)
            if len(q1.split(' ')) >= MAX_LENGTH:
                q1 = str(key)
            if len(a.split(" ")) >= MAX_LENGTH:
                a = re.split(r'[.?!]', a)[0]
            modified_ques.append(q1)
            modified_ans.append(a)
        else:
            q = qdict[key][ques]
            q1 = normalizeString(q)
            if len(q1.split(' ')) >= MAX_LENGTH:
                q1 = str(key)
            a1 = get_answer_remaining(qdict[key][ans])
            if a1:
                modified_ques.append(q1)
                modified_ans.append(a1)

    qa_dict['question'] = modified_ques
    qa_dict['answer'] = modified_ans
    with open('ques_ans.pickle', 'wb') as handle:
        pickle.dump(qa_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
