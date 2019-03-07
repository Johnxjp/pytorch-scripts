import torch.nn as nn


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size)

    def forward(self, x):
        return self.rnn(x)
