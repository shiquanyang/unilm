import torch
import torch.nn as nn
from models.BertEncoder import BertEncoder
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from s2s_ft.modeling import BertLayer
from models.ContextRNN import ContextRNN
from models.GraphMemory import GraphMemory
from models.VisualMemory import VisualMemory
from utils.config import *
import numpy as np
import pdb


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
                 visual_hop,
                 config):
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
        self.bert_encoder = BertEncoder(input_size, hidden_size, dropout)
        self.bert_layer = BertLayer(config)
        # self.context_rnn = ContextRNN(input_size, hidden_size, dropout)
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

    def gen_input_mask(self, batch_size, max_len, lengths):
        input_mask = np.zeros([batch_size, max_len], dtype=np.float32)
        for id, len in enumerate(lengths):
            input_mask[id, :int(lengths[id])] = np.ones([1, int(lengths[id])], dtype=np.float32)
        return torch.tensor(input_mask)

    def forward(self, story, conv_lens, cls_ids, input_turns, kb_arr, img_arr):
        turns = [turn.item() for turn in input_turns]
        output_turns = [turn.item() + 1 for turn in input_turns]
        max_turn = max(turns) + 1
        batch_size = story.size()[1]

        # BERT encoding
        dh_outputs, dh_hidden = self.bert_encoder(story, conv_lens)  # dh_outputs: batch_size * max_len * emb_dim.

        # initialize graph memory and visual memory
        self.graph_memory.load_graph(kb_arr)
        self.visual_memory.load_images(img_arr)

        # attend external knowledge
        query_vectors = torch.zeros([dh_hidden.size()[1], int(max_turn), dh_hidden.size()[2]])
        if USE_CUDA:
            query_vectors = query_vectors.cuda()
        for idx, ele in enumerate(cls_ids):
            for pos, val in enumerate(ele):
                query_vectors[idx, pos, :] = dh_outputs[idx, val, :]
        knowledge_vecs = torch.zeros([int(max_turn), dh_hidden.size()[1], 3 * dh_hidden.size()[2]])
        if USE_CUDA:
            knowledge_vecs = knowledge_vecs.cuda()
        for turn in range(int(max_turn)):
            kb_readout = self.graph_memory(query_vectors[:, turn, :])  # kb_readout: (batch_size*max_len) * embed_dim.
            vis_readout = self.visual_memory(query_vectors[:, turn, :])  # vis_readout: (batch_size*max_len) * embed_dim.
            knowledge_vecs[turn, :, :] = torch.cat([query_vectors[:, turn, :], kb_readout, vis_readout], dim=1)
        knowledge_vecs_t = knowledge_vecs.transpose(0, 1)
        knowledge_vecs_linear = self.Linear(knowledge_vecs_t)

        # inter-turn Transformer layer
        weight = self.gen_input_mask(batch_size, int(max_turn), output_turns)
        from_weight = weight.unsqueeze(-1)
        to_weight = weight.unsqueeze(1)
        from_weight_expand = from_weight.expand([from_weight.size()[0], from_weight.size()[1], from_weight.size()[1]])
        to_weight_expand = to_weight.expand([to_weight.size()[0], to_weight.size()[2], to_weight.size()[2]])
        attention_mask = from_weight_expand * to_weight_expand
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        if USE_CUDA:
            extended_attention_mask = extended_attention_mask.cuda()
        # pdb.set_trace()
        outputs = self.bert_layer(knowledge_vecs_linear, extended_attention_mask)

        return outputs[0], output_turns  # outputs: batch_size * max_turns * 768, turns: batch_size.
