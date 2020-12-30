import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class GlobalSemanticsAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, alpha, dropout):
        super(GlobalSemanticsAggregator, self).__init__()
        self.window_sizes = [1, 2, 3]
        # self.window_sizes = [1]
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.projector = nn.Linear(input_dim, output_dim)
        self.W = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(input_dim, output_dim).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a1 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(output_dim, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a2 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(output_dim, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def self_attention_layer(self, pooled_vectors, mask_matrix, window_cnt):
        batch_size = pooled_vectors.shape[0]
        # interacted_vectors = []
        # vectors = pooled_vectors[bt]  # vectors: window_cnt * embedding_dim.
        # h = torch.mm(pooled_vectors, self.W)
        interacted_vectors = torch.zeros(batch_size, self.output_dim).float()
        for bt in range(batch_size):
            input = pooled_vectors[bt]  # include pad information here.
            window_cnt_actual = int(window_cnt[bt].item())
            window_cnt_max = pooled_vectors.shape[1]
            h = torch.mm(input, self.W)
            f_1 = h @ self.a1
            f_2 = h @ self.a2
            e = self.leakyrelu(f_1 + f_2.transpose(0, 1))

            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(mask_matrix[bt] > 0, e, zero_vec)  # filter pad information here.
            attention = F.softmax(attention, dim=1)
            # # mask padded turn's attention to zero.
            # attention_padding = torch.zeros((window_cnt_max - window_cnt_actual), window_cnt_max).float()
            # attention[window_cnt_actual:, :] = attention_padding

            # remove attention dropout to deal with 1-turn dialogue.
            # attention = F.dropout(attention, self.dropout, training=self.training)
            h_prime = torch.matmul(attention, h)
            # average vectors
            # need to mask before average!!! Bug here!!!
            if window_cnt_actual != 0:
                h_prime_actual = h_prime[:window_cnt_actual, :]
            else:
                h_prime_actual = h_prime
            h_prime_avg = h_prime_actual.mean(0)
            # h_prime_avg = h_prime.mean(0)

            interacted_vectors[bt, :] = h_prime_avg
        # interacted_vectors.append(h_prime)
        return interacted_vectors  # interacted_vectors: batch_size * window_cnt * embedding_dim.

    def pooling_split_vectors(self, splitted_vectors):
        batch_size, window_cnt, embed_dim = splitted_vectors.shape[0], splitted_vectors.shape[1], splitted_vectors.shape[3]
        pooled_vectors = torch.zeros(batch_size, window_cnt, embed_dim).float()
        for bt in range(batch_size):
            vectors = splitted_vectors[bt]  # what if vectors is null?
            vectors_new = vectors.mean(1)  # average within window.
            # vectors_new = []
            # if len(vectors) != 0:
            #     for i in len(vectors):
            #         pooling_vector = np.mean(vectors[i], axis=0)  # calculate mean along col.
            #         vectors_new.append(pooling_vector)
            # pooled_vectors.append(vectors_new)  # vectors_new: window_cnt * embedding_dim.
            pooled_vectors[bt, :, :] = vectors_new
        return pooled_vectors  # pooled_vectors: batch_size * window_cnt * embedding_dim.

    def generate_mask_matrix(self, input_turns, window_cnt):
        batch_size = window_cnt.shape[0]
        turns = [turn.item() for turn in input_turns]  # turns: batch_size.
        max_turn = int(max(turns)) + 1
        mask_matrix = torch.zeros(batch_size, max_turn, max_turn).float()
        for bt in range(batch_size):
            node_cnt = int(window_cnt[bt].item())
            ones = torch.ones(node_cnt, node_cnt).float()
            mask_matrix[bt, :node_cnt, :node_cnt] = ones
        return mask_matrix

    def split_local_semantic_vectors(self, local_semantic_vectors, input_turns, window_size):
        local_semantic_vectors = local_semantic_vectors.transpose(0, 1)  # local_semantic_vectors: batch_size * turns * (3*embedding_dim).
        batch_size, embed_dim = local_semantic_vectors.shape[0], local_semantic_vectors.shape[2]
        turns = [turn.item() for turn in input_turns]  # turns: batch_size.
        max_turn = int(max(turns)) + 1
        splitted_vectors = torch.zeros(batch_size, max_turn, window_size, embed_dim).float()
        window_cnt = torch.zeros(batch_size).float()
        for bt in range(batch_size):
            vectors = torch.zeros(max_turn, window_size, embed_dim).float()
            t = 0
            while window_size + t - 1 <= turns[bt]:
                # tmp = torch.zeros(window_size, embed_dim).float()
                tmp = local_semantic_vectors[bt, t:(t+window_size), :]
                # for step in range(window_size):
                #     tmp.append(local_semantic_vectors[bt][t+step])
                # vectors.append(tmp)
                vectors[t, :, :] = tmp
                t += 1
            # splitted_vectors.append(vectors)
            # window_cnt.append(t)
            splitted_vectors[bt, :, :, :] = vectors
            window_cnt[bt] = t
        return splitted_vectors, window_cnt  # splitted_vectors: batch_size * window_cnt * window_size, window_cnt: batch_size.

    def compute_valid_window_size(self, batch_size, window_cnt_list):
        ret = []
        for bt in range(batch_size):
            t = 0
            for element in window_cnt_list:
                if element[bt] > 0.0:
                    t += 1
            ret.append(t)
        return ret

    def forward(self, local_semantic_vectors, input_turns):  # local_semantic_vectors: turns * batch_size * (3*embedding_dim), input_turns: batch_size * 1.
        # ngram_vectors = []  # ngram_vectors: window_num * batch_size * embedding_dim.
        batch_size, embed_dim = local_semantic_vectors.shape[1], local_semantic_vectors.shape[2]
        window_num = len(self.window_sizes)
        window_cnt_list = []
        ngram_vectors = torch.zeros(window_num, batch_size, self.output_dim).float()
        for idx, window_size in enumerate(self.window_sizes):
            splitted_vectors, window_cnt = self.split_local_semantic_vectors(local_semantic_vectors, input_turns, window_size)
            window_cnt_list.append(window_cnt.tolist())
            pooled_vectors = self.pooling_split_vectors(splitted_vectors)
            mask_matrix = self.generate_mask_matrix(input_turns, window_cnt)
            interacted_vectors = self.self_attention_layer(pooled_vectors, mask_matrix, window_cnt)
            ngram_vectors[idx, :, :] = interacted_vectors
        # compute average per-sample because valid window_size is different.
        valid_window_size = self.compute_valid_window_size(batch_size, window_cnt_list)
        average_merged_ngram_vectors = torch.zeros(batch_size, embed_dim).float()
        for bt in range(batch_size):
            temp = ngram_vectors[:valid_window_size[bt], bt, :]
            average_merged_ngram_vectors[bt, :] = temp.mean(0)
        # average_merged_ngram_vectors = ngram_vectors.transpose(0, 1).mean(1)
        return average_merged_ngram_vectors
