import torch


def one_hot(x, categories):
    """
    Transforms a tensor of integer labels into a one-hot vector

    :param x: 1d Tensor of integer labels
    :param categories: number of classes
    :return: one-hot tensor where each row is a one-hot tensor of the label at
    the same index position in x
    """

    result = torch.zeros((len(x), categories), dtype=x.dtype)
    for i in range(categories):
        result[:, i][x == i] = 1

    return result


def reverse_one_hot(x):
    return torch.argmax(x, dim=1)


def normalise(x):

    if len(x.size()) == 1:
        return x / torch.norm(x)

    return x / torch.norm(x, dim=1, keepdim=True)
