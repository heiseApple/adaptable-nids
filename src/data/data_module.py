import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader, TensorDataset

from util.config import load_config


class DataModule:
    def __init__(self, train_dataset, val_dataset, test_dataset=None, **kwargs):
        cf = load_config()
        self.batch_size = kwargs.get('batch_size', cf['batch_size'])
        self.adapt_batch_size = kwargs.get('adapt_batch_size', cf['adapt_batch_size'])
        self.num_workers = kwargs.get('num_workers', cf['num_workers'])
        self.pin_memory = kwargs.get('pin_memory', cf['pin_memory'])

        self.train_x, self.train_y = train_dataset
        self.val_x, self.val_y = val_dataset
        self.test_x, self.test_y = test_dataset or (None, None)
        
        self.approach_type = kwargs.get('appr_type', None)
        self.seed = kwargs.get('seed', cf['seed'])
        
        
    @staticmethod
    def add_argparse_args(parent_parser):
        cf = load_config()
        parser = ArgumentParser(
            parents=[parent_parser],
            add_help=True,
            conflict_handler='resolve',
        )
        parser.add_argument('--batch-size', type=int, default=cf['batch_size'])
        parser.add_argument('--adapt-batch-size', type=int, default=cf['adapt_batch_size'])
        parser.add_argument('--num-workers', type=int, default=cf['num_workers'])
        parser.add_argument('--pin-memory', action='store_true', default=cf['pin_memory'])
        return parser
    
    def set_train_dataset(self, dataset):
        self.train_x, self.train_y = dataset
        
    def set_val_dataset(self, dataset):
        self.val_x, self.val_y = dataset
        
    def set_test_dataset(self, dataset):
        self.test_x, self.test_y = dataset

    def get_train_data(self):
        
        if self.approach_type == 'dl':
            data_tensor = torch.from_numpy(self.train_x).float()
            labels_tensor = torch.from_numpy(self.train_y).float()
            
            return DataLoader(
                TensorDataset(data_tensor, labels_tensor),
                batch_size=self.batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(self.seed),
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        else:
            # For ML approaches
            return self.train_x, self.train_y
        
        
    def get_val_data(self):
        
        if self.approach_type == 'dl':
            data_tensor = torch.from_numpy(self.val_x).float()
            labels_tensor = torch.from_numpy(self.val_y).float()
            
            return DataLoader(
                TensorDataset(data_tensor, labels_tensor),
                batch_size=self.batch_size,
                shuffle=False,
                generator=torch.Generator().manual_seed(self.seed),
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        else:
            # For ML approaches
            return self.val_x, self.val_y   
        
        
    def get_test_data(self):
        
        if self.approach_type == 'dl':
            data_tensor = torch.from_numpy(self.test_x).float()
            labels_tensor = torch.from_numpy(self.test_y).float()
            
            return DataLoader(
                TensorDataset(data_tensor, labels_tensor),
                batch_size=self.batch_size,
                shuffle=False,
                generator=torch.Generator().manual_seed(self.seed),
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        else:
            # For ML approaches
            return self.test_x, self.test_y
        
        
    def get_adapt_data(self):
        
        if self.approach_type == 'dl':
            data_tensor = torch.from_numpy(self.train_x).float()
            labels_tensor = torch.from_numpy(self.train_y).float()
            
            return DataLoader(
                TensorDataset(data_tensor, labels_tensor),
                batch_size=self.adapt_batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(self.seed),
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        else:
            # For ML approaches
            return self.train_x, self.train_y
        