import torch
import torch.nn as nn
from utils.utils_temp import AttrProxy


class VisualMemory(nn.Module):
    def __init__(self, input_channels, output_channels, conv_kernel_size, pool_kernel_size, embedding_dim, hop, dropout):
        super(VisualMemory, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv_kernel_size = conv_kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.max_hops = hop
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        for hop in range(self.max_hops + 1):
            C = nn.Conv2d(input_channels, output_channels, conv_kernel_size)  # need to add BN and Relu layer.
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d(pool_kernel_size)  # need to add a Flatten() layer ?
        self.fc = nn.Linear(2304, embedding_dim)  # need to check this logic and make sure the pool_output_size.
        self.softmax = nn.Softmax(dim=1)

    # def load_images(self, image_arr, hidden):
    #     u = [hidden.squeeze(0)]
    #     image_arr_size = image_arr.size()
    #     batch_size = int(image_arr_size[0])
    #     self.m_image = []
    #     for hop in range(self.max_hops):
    #         embed_A = self.C[hop](image_arr.contiguous().view([image_arr_size[0]*image_arr_size[1]] + list(image_arr_size[2:])))  # need to check the logic clearly.
    #         embed_A = self.relu(embed_A)
    #         embed_A = self.avgpool(embed_A)
    #         embed_A = torch.flatten(embed_A, 1)
    #         embed_A = self.fc(embed_A)
    #         embed_A = embed_A.view([batch_size, int(embed_A.size()[0]/batch_size)] + list(embed_A.size()[1:]))
    #         # embed_A = embed_A.view((image_arr_size[0], image_arr_size[1]) + embed_A.size()[1:])
    #         embed_A = self.dropout_layer(embed_A)  # need to add dropout layer ?
    #
    #         if (len(list(u[-1].size())) == 1):
    #             u[-1] = u[-1].unsqueeze(0)  ## used for bsz = 1.
    #         u_temp = u[-1].unsqueeze(1).expand_as(embed_A)
    #         # u_temp = u[-1].expand_as(embed_A)
    #         prob_logit = torch.sum(embed_A * u_temp, 2)
    #         prob_ = self.softmax(prob_logit)
    #
    #         embed_C = self.C[hop+1](image_arr.contiguous().view([image_arr_size[0]*image_arr_size[1]] + list(image_arr_size[2:])))
    #         embed_C = self.relu(embed_C)
    #         embed_C = self.avgpool(embed_C)
    #         embed_C = torch.flatten(embed_C, 1)
    #         embed_C = self.fc(embed_C)
    #         embed_C = embed_C.view([batch_size, int(embed_C.size()[0]/batch_size)] + list(embed_C.size()[1:]))
    #         prob = prob_.unsqueeze(2).expand_as(embed_C)
    #         o_k = torch.sum(embed_C * prob, 1)
    #         u_k = u[-1] + o_k
    #         u.append(u_k)
    #         self.m_image.append(embed_A)
    #     self.m_image.append(embed_C)
    #     return u[-1]

    def load_images(self, image_arr):
        image_arr_size = image_arr.size()
        batch_size = int(image_arr_size[0])
        self.m_image = []
        for hop in range(self.max_hops):
            embed_A = self.C[hop](image_arr.contiguous().view([image_arr_size[0]*image_arr_size[1]] + list(image_arr_size[2:])))  # need to check the logic clearly.
            embed_A = self.relu(embed_A)
            embed_A = self.avgpool(embed_A)
            embed_A = torch.flatten(embed_A, 1)
            embed_A = self.fc(embed_A)
            embed_A = embed_A.view([batch_size, int(embed_A.size()[0]/batch_size)] + list(embed_A.size()[1:]))
            # embed_A = embed_A.view((image_arr_size[0], image_arr_size[1]) + embed_A.size()[1:])
            embed_A = self.dropout_layer(embed_A)  # need to add dropout layer ?

            embed_C = self.C[hop+1](image_arr.contiguous().view([image_arr_size[0]*image_arr_size[1]] + list(image_arr_size[2:])))
            embed_C = self.relu(embed_C)
            embed_C = self.avgpool(embed_C)
            embed_C = torch.flatten(embed_C, 1)
            embed_C = self.fc(embed_C)
            embed_C = embed_C.view([batch_size, int(embed_C.size()[0]/batch_size)] + list(embed_C.size()[1:]))
            self.m_image.append(embed_A)
        self.m_image.append(embed_C)
        return image_arr_size

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
            m_A = self.m_image[hop]
            if (len(list(u[-1].size())) == 1):
                u[-1] = u[-1].unsqueeze(0)  ## used for bsz = 1.
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            if USE_CUDA:
                u_temp = u_temp.cuda()
            prob_logits = torch.sum(m_A * u_temp, 2)
            prob_soft = self.softmax(prob_logits)
            m_C = self.m_image[hop + 1]
            prob = prob_soft.unsqueeze(2).expand_as(m_C)
            o_k = torch.sum(m_C * prob, 1)
            if USE_CUDA:
                u[-1] = u[-1].cuda()
            u_k = u[-1] + o_k
            u.append(u_k)
        return u[-1]