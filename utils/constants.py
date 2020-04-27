# This file contains all the constant variables required by the project
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CSV_PATH = "saved_files/20200325_counsel_chat.csv"
MODEL_STATE_PATH = "saved_files/model_state_dict"
glove_path = 'saved_files/glove_vec_emb.pkl'
weight_matrix_path = 'saved_files/weight_matrix.pkl'
vocab_dict_path = "saved_files/vocab_dict.pkl"
MAX_LENGTH = 72
hidden_size = 128
ques_ans_data_path = "saved_files/ques_ans.pickle"
ques = 'question'
ans = 'answer'
SOS_token = 0
EOS_token = 1
glove_model_path = 'Downloads'