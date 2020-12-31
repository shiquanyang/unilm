import torch
import torch.nn as nn
from utils.config import *
from utils.utils_temp import AttrProxy
import pdb


class GraphMemory(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, dropout):
        super(GraphMemory, self).__init__()
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        for hop in range(self.max_hops + 1):
            C = nn.Embedding(vocab, embedding_dim, padding_idx=PAD_token)
            C.weight.data.normal_(0, 0.1)
            # t = torch.randn(vocab, embedding_dim) * 0.1
            # t[PAD_token, :] = torch.zeros(1, embedding_dim)
            # C.weight.data = t
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.conv_layer = nn.Conv1d(embedding_dim, embedding_dim, 5, padding=2)

    # def load_graph(self, story, hidden):
    #     # Forward multiple hop mechanism
    #     u = [hidden.squeeze(0)]
    #     story = story.transpose(0, 1)
    #     story_size = story.size()
    #     self.m_story = []
    #     for hop in range(self.max_hops):
    #         embed_A = self.C[hop](story.contiguous().view(story_size[0], -1))  # .long()) # b * (m * s) * e
    #         embed_A = embed_A.view(story_size + (embed_A.size(-1),))  # b * m * s * e
    #         embed_A = torch.sum(embed_A, 2).squeeze(2)  # b * m * e
    #         embed_A = self.dropout_layer(embed_A)
    #
    #         if (len(list(u[-1].size())) == 1):
    #             u[-1] = u[-1].unsqueeze(0)  ## used for bsz = 1.
    #         u_temp = u[-1].unsqueeze(1).expand_as(embed_A)
    #         prob_logit = torch.sum(embed_A * u_temp, 2)
    #         prob_ = self.softmax(prob_logit)
    #
    #         embed_C = self.C[hop + 1](story.contiguous().view(story_size[0], -1).long())
    #         embed_C = embed_C.view(story_size + (embed_C.size(-1),))
    #         embed_C = torch.sum(embed_C, 2).squeeze(2)
    #
    #         prob = prob_.unsqueeze(2).expand_as(embed_C)
    #         o_k = torch.sum(embed_C * prob, 1)
    #         u_k = u[-1] + o_k
    #         u.append(u_k)
    #         self.m_story.append(embed_A)
    #     self.m_story.append(embed_C)
    #     return u[-1]

    def load_graph(self, story):
        # Forward multiple hop mechanism
        story = story.transpose(0, 1)
        story_size = story.size()
        self.m_story = []
        for hop in range(self.max_hops):
            embed_A = self.C[hop](story.contiguous().view(story_size[0], -1))  # .long()) # b * (m * s) * e
            embed_A = embed_A.view(story_size + (embed_A.size(-1),))  # b * m * s * e
            embed_A = torch.sum(embed_A, 2).squeeze(2)  # b * m * e
            embed_A = self.dropout_layer(embed_A)

            embed_C = self.C[hop + 1](story.contiguous().view(story_size[0], -1).long())
            embed_C = embed_C.view(story_size + (embed_C.size(-1),))
            embed_C = torch.sum(embed_C, 2).squeeze(2)

            self.m_story.append(embed_A)
        self.m_story.append(embed_C)
        return story_size

    # def forward(self, query_vector):
    #     u = [query_vector]
    #     for hop in range(self.max_hops):
    #         m_A = self.m_story[hop]
    #         if (len(list(u[-1].size())) == 1):
    #             u[-1] = u[-1].unsqueeze(0)  ## used for bsz = 1.
    #         u_temp = u[-1].unsqueeze(1).expand_as(m_A)
    #         prob_logits = torch.sum(m_A * u_temp, 2)
    #         prob_soft = self.softmax(prob_logits)
    #         m_C = self.m_story[hop + 1]
    #         prob = prob_soft.unsqueeze(2).expand_as(m_C)
    #         o_k = torch.sum(m_C * prob, 1)
    #         u_k = u[-1] + o_k
    #         u.append(u_k)
    #     return prob_soft, prob_logits

    def forward(self, query_vector):
        u = [query_vector.squeeze(0)]
        for hop in range(self.max_hops):
            m_A = self.m_story[hop]
            if (len(list(u[-1].size())) == 1):
                u[-1] = u[-1].unsqueeze(0)  ## used for bsz = 1.
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            # pdb.set_trace()
            if USE_CUDA:
                u_temp = u_temp.cuda()
            prob_logits = torch.sum(m_A * u_temp, 2)
            prob_soft = self.softmax(prob_logits)
            m_C = self.m_story[hop + 1]
            prob = prob_soft.unsqueeze(2).expand_as(m_C)
            o_k = torch.sum(m_C * prob, 1)
            if USE_CUDA:
                u[-1] = u[-1].cuda()
            u_k = u[-1] + o_k
            u.append(u_k)
        return u[-1]
