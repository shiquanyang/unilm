import torch
import torch.nn as nn


class Calibration(nn.Module):
    def __init__(self, vocab, embedding_dim):
        super(Calibration, self).__init__()
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        # self.projector = nn.Linear(2*embedding_dim, vocab)
        self.projector = nn.Linear(3*embedding_dim, vocab)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        t = self.projector(input)
        t1 = self.sigmoid(t)
        return self.sigmoid(self.projector(input))