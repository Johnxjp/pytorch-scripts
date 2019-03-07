import torch
import torch.nn as nn


class BasicFeedForward(nn.Module):

    def __init__(self, d_in, d_out):
        """
        :param d_in: input dimensions
        :param d_out: output dimensions = n_classes
        """
        super().__init__()
        self.out = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.out(x)


class MultilayerFeedForward(nn.Module):

    def __init__(self, d_in, d_out, hidden_dims=None,
                 activation_function=nn.ReLU):
        """
        :param d_in: input dimensions
        :param d_out: output dimensions
        :param hidden_dims: list of hidden layer input dimensions
        :param activation_function: activation function
        """

        super().__init__()
        self.activation = activation_function

        self.layers = nn.ModuleList()
        if not hidden_dims:
            for h_out in hidden_dims:
                self.layers.append(nn.Linear(d_in, h_out))
                d_in = h_out

        self.out = nn.Linear(d_in, d_out)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)

        return self.out(x)


class DropoutFeedForward(nn.Module):

    def __init__(self, d_in, d_out, hidden_dims=None,
                 activation_function=nn.ReLU, dropout=0):
        """
        :param d_in: input dimensions
        :param d_out: output dimensions
        :param hidden_dims: list of hidden layer input dimensions
        :param activation_function: activation function
        """

        super().__init__()
        self.activation = activation_function
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList()
        if not hidden_dims:
            for h_out in hidden_dims:
                self.layers.append(nn.Linear(d_in, h_out))
                d_in = h_out

        self.out = nn.Linear(d_in, d_out)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)

        return self.out(x)


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size)

    def forward(self, x):
        return self.rnn(x)
