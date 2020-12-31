import torch
import torch.nn as nn
from utils.config import *
from transformers import BertModel
import numpy as np
import pdb


class BertEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, n_layers=1):
        super(BertEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=PAD_token)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.W1 = nn.Linear(768, hidden_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def gen_input_mask(self, batch_size, max_len, lengths):
        input_mask = np.zeros([batch_size, max_len], dtype=np.float32)
        for id, len in enumerate(lengths):
            input_mask[id, :lengths[id]] = np.ones([1, lengths[id]], dtype=np.float32)
        return torch.tensor(input_mask)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # pdb.set_trace()
        # print(self.bert.device)
        max_len = input_seqs.shape[0]
        mask = self.gen_input_mask(input_seqs.shape[1], input_seqs.shape[0], input_lengths)
        if USE_CUDA:
            mask = mask.to(self.device)
            input_seqs = input_seqs.transpose(0, 1).type(torch.LongTensor).cuda()
        else:
            input_seqs = input_seqs.transpose(0, 1).type(torch.LongTensor)
        outputs = self.bert(input_seqs, attention_mask=mask)
        last_hidden_states = outputs[0]
        hidden = torch.sum(last_hidden_states, dim=1) / max_len
        hidden = self.W1(hidden.unsqueeze(0))
        outputs = self.W1(last_hidden_states)
        return outputs, hidden






