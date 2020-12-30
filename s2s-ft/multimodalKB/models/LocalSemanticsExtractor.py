import torch
import torch.nn as nn
from models.BertEncoder import BertEncoder
from models.ContextRNN import ContextRNN
from models.GraphMemory import GraphMemory
from models.VisualMemory import VisualMemory
from utils.config import *
import numpy as np


class LocalSemanticsExtractor(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 dropout,
                 lang,
                 vocab,
                 embedding_dim,
                 graph_hop,
                 input_channels,
                 output_channels,
                 conv_kernel_size,
                 pool_kernel_size,
                 visual_hop):
        super(LocalSemanticsExtractor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.lang = lang
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.graph_hop = graph_hop
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv_kernel_size = conv_kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.visual_hop = visual_hop
        # self.bert_encoder = BertEncoder(input_size, hidden_size, dropout)
        self.context_rnn = ContextRNN(input_size, hidden_size, dropout)
        self.graph_memory = GraphMemory(vocab, embedding_dim, graph_hop, dropout)
        self.visual_memory = VisualMemory(input_channels, output_channels, conv_kernel_size,
                                          pool_kernel_size, embedding_dim, visual_hop, dropout)
        self.Linear = nn.Linear(3 * embedding_dim, 768)

    def pad_local_dialogues(self, local_dialogues, lengths):
        max_turn = len(local_dialogues)
        padded_local_dialogues = []
        for turn in range(max_turn):
            dialogues = local_dialogues[turn]
            lens = lengths[turn]
            max_len = max(lens)
            padded_seqs = np.ones([len(dialogues), max_len, MEM_TOKEN_SIZE])
            for i, seq in enumerate(dialogues):
                end = lens[i]
                # if len(seq) != 0:
                if end != 0:
                    t = seq[:end]
                    padded_seqs[i, :end, :] = seq[:end]
            padded_local_dialogues.append(padded_seqs)
        return padded_local_dialogues

    def extract_dialogue_by_turn(self, conv_arr, turn):
        ret = []
        user_encoding = self.lang.word2index['$u']
        turn_encoding = self.lang.word2index['turn'+str(turn)]
        for arr in conv_arr:
            word, speaker, turn_num, word_num = arr[0], arr[1], arr[2], arr[3]
            # if int(turn_num.split('turn')[1]) < int(turn) \
            #         or (int(turn_num.split('turn')[1]) == int(turn) and speaker == '$u'):
            # if (int(turn_num) < int(turn_encoding) and int(turn_num) != 1) \
            #         or (int(turn_num) == int(turn_encoding) and int(speaker) == int(user_encoding)):
            if int(turn_num) == int(turn_encoding):
                ret.append(arr.numpy())
        return ret

    def has_turn(self, conv_arr, turn):
        turn_encoding = self.lang.word2index['turn'+str(turn)]
        for arr in conv_arr:
            if turn_encoding in arr:
                return True
        return False

    def generate_local_dialogues(self, story, max_turn):
        global_dialogues, global_lengths, global_pseudo_lengths = [], [], []
        batch_size = story.shape[1]
        story = story.transpose(0, 1)
        for t in range(int(max_turn)):
            local_dialogues, lengths, pseudo_lengths = [], [], []
            for bt in range(batch_size):
                if self.has_turn(story[bt], t):
                    ret = self.extract_dialogue_by_turn(story[bt], t)
                    length = len(ret)
                    pseudo_length = len(ret)
                else:
                    ret = [[PAD_token] * MEM_TOKEN_SIZE]
                    length = 0
                    pseudo_length = 1  # for context_rnn use.
                local_dialogues.append(ret)
                lengths.append(length)
                pseudo_lengths.append(pseudo_length)
            global_dialogues.append(local_dialogues)
            global_lengths.append(lengths)
            global_pseudo_lengths.append(pseudo_lengths)
        return global_dialogues, global_lengths, global_pseudo_lengths

    def sort_input(self, input_seqs, sorted_lengths):
        ret = [input_seqs[idx] for idx in sorted_lengths]
        ret = torch.Tensor(ret).long()
        return ret

    def desort_output(self, output_seqs, sorted_lengths):
        unpacked_seq_lengths = np.argsort(sorted_lengths)
        ret = torch.zeros(output_seqs.size()[0], output_seqs.size()[1]).float()
        for idx, item in enumerate(unpacked_seq_lengths):
            ret[idx, :] = output_seqs[item, :]
        return ret

    def mask_results(self, encoded_hidden, input_lengths):
        ret = torch.zeros(encoded_hidden.size()[0], encoded_hidden.size()[1]).float()
        for idx, length in enumerate(input_lengths):
            if length != 0:
                ret[idx, :] = encoded_hidden[idx, :]
        return ret

    def forward(self, story, input_turns, kb_arr, img_arr):
        turns = [turn.item() for turn in input_turns]
        output_turns = [turn.item() + 1 for turn in input_turns]
        max_turn = max(turns) + 1
        batch_size = story.size()[1]
        # parse local dialogue data and padding.
        local_dialogues, lengths, pseudo_lengths = self.generate_local_dialogues(story, max_turn)  #  lengths: turns * batch_size.
        padded_local_dialogues = self.pad_local_dialogues(local_dialogues, lengths)
        # initialize graph memory and visual memory.
        self.graph_memory.load_graph(kb_arr)
        self.visual_memory.load_images(img_arr)
        # iteratively compute local semantic vectors.
        # local_semantic_vectors = []
        # local_semantic_vectors = torch.zeros(int(max_turn), batch_size, 2 * self.embedding_dim)
        local_semantic_vectors = torch.zeros(int(max_turn), batch_size, 3 * self.embedding_dim)
        for turn in range(int(max_turn)):
            input_seqs = padded_local_dialogues[turn]
            # use pseudo lengths for context_rnn.
            input_lengths = lengths[turn]
            pseudo_input_lengths = pseudo_lengths[turn]
            # index of samples within a batch in decreasing-order according to pseudo_lengths.
            sorted_lengths = np.argsort(pseudo_input_lengths)[::-1]
            # sort input_seqs according to the index.
            sorted_input_seqs = self.sort_input(input_seqs, sorted_lengths)
            # sort input_lengths according to the index.
            sorted_input_lengths = np.sort(pseudo_input_lengths).tolist()[::-1]
            dh_outputs, dh_hidden = self.context_rnn(sorted_input_seqs.transpose(0, 1), sorted_input_lengths)
            encoded_hidden = self.desort_output(dh_hidden.squeeze(0), sorted_lengths)
            kb_readout = self.graph_memory(encoded_hidden)
            vis_readout = self.visual_memory(encoded_hidden)
            encoded_hidden = torch.cat((encoded_hidden, kb_readout, vis_readout), dim=1)
            # encoded_hidden = torch.cat((encoded_hidden, vis_readout), dim=1)
            # mask pure-padded samples according to actual lengths.
            encoded_hidden = self.mask_results(encoded_hidden, input_lengths)
            local_semantic_vectors[turn, :, :] = encoded_hidden   # local_semantic_vectors: turns * batch_size * (3*embedding_dim).
            # local_semantic_vectors.append(encoded_hidden)
        # # use simple average to merge different-turns local_semantic_vectors instead of GlobalSemanticsAggregator.
        # final_local_semantic_vectors = torch.zeros(batch_size, 3 * self.embedding_dim)
        # for idx, turn in enumerate(turns):
        #     temp = torch.zeros((int(turn) + 1), 3 * self.embedding_dim)
        #     for i in range(int(turn) + 1):
        #         temp[i, :] = local_semantic_vectors[i, idx, :]
        #     final_local_semantic_vectors[idx, :] = temp.mean(0)
        # extract final-turn vectors for simulate baseline
        # final_local_semantic_vectors = torch.zeros(batch_size, 3 * self.embedding_dim).float()
        # for idx, turn in enumerate(turns):
        #     final_local_semantic_vectors[idx, :] = local_semantic_vectors[int(turn), idx, :]
        # return final_local_semantic_vectors, lengths  # local_semantic_vectors: max_turns * batch_size * (3 * embedding_dim), lengths: max_turns * batch_size.
        local_semantic_vectors = self.Linear(local_semantic_vectors.transpose(0, 1))
        return local_semantic_vectors, output_turns  # local_semantic_vectors: max_turns * batch_size * (3 * embedding_dim), turns: batch_size.
