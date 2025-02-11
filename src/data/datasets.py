# from torch.utils.data import Dataset as TorchDataset

# class Dataset(TorchDataset):
#     def __init__(self, data, targets):        
#         self.data, self.targets = data, targets

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         x = self.data[index]
#         y = self.targets[index]
#         return x, y