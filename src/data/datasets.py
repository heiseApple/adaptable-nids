import os
from torch.utils.data import Dataset as TorchDataset

class Dataset(TorchDataset):
    def __init__(self, root):
        self.root = os.path.expanduser(root)
        
        self.data, self.targets = self.load_data()

    def load_data(self):
        
        return data, targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y