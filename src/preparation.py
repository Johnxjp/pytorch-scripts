from torch.utils import data


class SimpleDataset(data.Dataset):
    """Simple Dataset class for loading data that fits in memory"""

    def __init__(self, x, y):
        """
        From data loaded into memory

        :param x: torch FloatTensor
        :param y: torch LongTensor
        """

        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
