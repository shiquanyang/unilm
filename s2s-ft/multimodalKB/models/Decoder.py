import torch
import torch.nn as nn
from utils.utils_general import _cuda
from utils.config import *
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn import init
import math


class Decoder(nn.Module):
    def __init__(self, shared_emb, lang, embedding_dim, dropout):
        super(Decoder, self).__init__()
        self.num_vocab = lang.n_words
        self.lang = lang
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.C = shared_emb
        self.softmax = nn.Softmax(dim=1)
        self.rnn = nn.GRU(embedding_dim, embedding_dim, dropout=dropout)
        self.relu = nn.ReLU()
        self.projector = nn.Linear(3*embedding_dim, embedding_dim)
        self.projector2 = nn.AdaptiveAvgPool2d((embedding_dim, 1))
        self.projector3 = nn.AdaptiveAvgPool3d((embedding_dim, 1, 1))
        # bilinear fusion
        self.weight = Parameter(torch.Tensor(embedding_dim, embedding_dim, embedding_dim))
        # trilinear fusion
        # self.weight = Parameter(torch.Tensor(embedding_dim, (embedding_dim+1), (embedding_dim+1), (embedding_dim+1)))
        self.bias = Parameter(torch.Tensor(embedding_dim))
        # init parameters
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)
        # self.projector = nn.Linear(2*embedding_dim, embedding_dim)
        # self.projector = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, encode_hidden, target_batches, max_target_length, batch_size, use_teacher_forcing, get_decoded_words, calibration_vocab):
        # Initialize variables for vocab and pointer
        all_decoder_outputs_vocab = _cuda(torch.zeros(max_target_length, batch_size, self.num_vocab))
        decoder_input = _cuda(torch.LongTensor([SOS_token] * batch_size))
        decoded_fine, decoded_coarse = [], []

        hidden = self.relu(self.projector(encode_hidden)).unsqueeze(0)
        # hidden = self.relu(self.projector2(encode_hidden).squeeze(2)).unsqueeze(0)
        # hidden = self.relu(self.projector3(encode_hidden).squeeze()).unsqueeze(0)
        # if batch_size == 1:
        #     hidden = hidden.unsqueeze(0)

        # bilinear fusion
        # encode_hidden_t = torch.zeros(batch_size, self.embedding_dim)
        # for bt, fea in enumerate(encode_hidden):
        #     for dim_out in range(self.embedding_dim):
        #         out = torch.sum(fea * self.weight[dim_out]) + self.bias[dim_out]
        #         encode_hidden_t[bt, dim_out] = out
        # hidden = self.relu(encode_hidden_t).unsqueeze(0).contiguous().cuda()
        # hidden = self.relu(encode_hidden_t).unsqueeze(0)

        # Start to genereate word-by-word
        for t in range(max_target_length):
            embed_q = self.dropout_layer(self.C(decoder_input))  # b * e
            if len(embed_q.size()) == 1: embed_q = embed_q.unsqueeze(0)
            _, hidden = self.rnn(embed_q.unsqueeze(0), hidden)

            p_vocab = self.attend_vocab(self.C.weight, hidden.squeeze(0))
            all_decoder_outputs_vocab[t] = p_vocab
            # calibration
            # p_vocab = self.softmax(p_vocab)
            # p_vocab = p_vocab
            # p_vocab = p_vocab + calibration_vocab
            _, topvi = p_vocab.data.topk(1)

            if use_teacher_forcing:
                decoder_input = target_batches[:, t]
            else:
                decoder_input = topvi.squeeze()

            if get_decoded_words:
                temp_f, temp_c = [], []
                for bi in range(batch_size):
                    token = topvi[bi].item()  # topvi[:,0][bi].item()
                    temp_c.append(self.lang.index2word[token])
                    temp_f.append(self.lang.index2word[token])
                decoded_fine.append(temp_f)
                decoded_coarse.append(temp_c)

        return all_decoder_outputs_vocab, decoded_fine, decoded_coarse

    def attend_vocab(self, seq, cond):
        scores_ = cond.matmul(seq.transpose(1,0))
        # scores = F.softmax(scores_, dim=1)
        return scores_