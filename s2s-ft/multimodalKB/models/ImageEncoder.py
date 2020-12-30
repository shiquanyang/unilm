import torch
import torch.nn as nn
from utils.utils_temp import AttrProxy


class ImageEncoder(nn.Module):
    def __init__(self, input_channels, output_channels, conv_kernel_size, pool_kernel_size, embedding_dim, hop, dropout):
        super(ImageEncoder, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv_kernel_size = conv_kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.max_hops = hop
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.C = nn.Conv2d(input_channels, output_channels, conv_kernel_size)  # need to add BN and Relu layer.
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d(pool_kernel_size)  # need to add a Flatten() layer ?
        self.fc = nn.Linear(2304, embedding_dim)  # need to check this logic and make sure the pool_output_size.
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image_arr, hidden):
        u = [hidden.squeeze(0)]
        image_arr_size = image_arr.size()
        batch_size = int(image_arr_size[0])
        embed_A = self.C(image_arr.contiguous().view([image_arr_size[0] * image_arr_size[1]] + list(image_arr_size[2:])))  # need to check the logic clearly.
        embed_A = self.relu(embed_A)
        embed_A = self.avgpool(embed_A)
        embed_A = torch.flatten(embed_A, 1)
        embed_A = self.fc(embed_A)
        embed_A = embed_A.view([batch_size, int(embed_A.size()[0] / batch_size)] + list(embed_A.size()[1:]))

        u_temp = u[-1].unsqueeze(1).expand_as(embed_A)
        prob_logit = torch.sum(embed_A * u_temp, 2)
        prob_ = self.softmax(prob_logit)

        prob = prob_.unsqueeze(2).expand_as(embed_A)
        ret = torch.sum(embed_A * prob, 1)
        # ret = torch.cat((embed_A[:, 0, :], embed_A[:, 1, :]), dim=1)
        return ret