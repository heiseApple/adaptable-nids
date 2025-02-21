import numpy as np
from sklearn.model_selection import train_test_split

from data.data_module import DataModule
from data.data_reader import get_data_labels
from util.config import load_config


class DataManager:
    """
    Manages and prepares data for training, validation, and testing.
    """
    def __init__(self, args):
        self.args = args
        self.src_dataset = None
        self.trg_dataset = None
        
        self.src_splits = None
        self.trg_splits = None
        self.num_classes = None
        
        self.appr_type = self.args.appr_type
        self.seed = self.args.seed
        
    def load_datasets(self):
        """
        Loads source and target datasets based on provided arguments.
        """
        dataset_args = {
            'num_pkts': self.args.num_pkts,
            'fields': self.args.fields,
            'is_flat': self.args.is_flat,
            'seed': self.args.seed,
        }
        self.src_dataset = get_data_labels(dataset=self.args.src_dataset, **dataset_args)
        if self.args.trg_dataset is not None:
            self.trg_dataset = get_data_labels(dataset=self.args.trg_dataset, **dataset_args)

    def split_dataset(self):
        """
        Splits the source dataset and, if present, also the target dataset.
        It also sets the number of classes and checks for compatibility.
        """
        self.src_splits, self.num_classes = self._train_val_test_split(dataset=self.src_dataset)
        if self.trg_dataset:
            self.trg_splits, trg_num_classes = self._train_val_test_split(dataset=self.trg_dataset)
            assert self.num_classes == trg_num_classes, (
                'Mismatch between the classes of the source and target datasets'
            )
        return self.src_splits, self.trg_splits, self.num_classes
    
    def get_datamodule(self, train, val, test=None):
        """Creates the DataModule from the provided splits"""
        return DataModule(
            train_dataset=train,
            val_dataset=val,
            test_dataset=test,
            appr_type=self.appr_type
        )

    def _train_val_test_split(self, dataset):
        cf = load_config()
        x = dataset['data']
        y = dataset['labels']
       
        # Train / Test
        x_train, x_test, y_train, y_test = train_test_split(
            x, y,
            train_size=cf['train_test_split'],
            random_state=self.seed,
            stratify=y
        )
        # Train / Val
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train,
            train_size=cf['train_val_split'],
            random_state=self.seed,
            stratify=y_train
        )
        return {
            'train': (x_train, y_train),
            'val': (x_val, y_val),
            'test': (x_test, y_test),
        }, len(np.unique(y))
        
    def concat_dataset(self, d1, d2):
        x1, y1 = d1
        x2, y2 = d2
        return np.concatenate([x1, x2]), np.concatenate([y1, y2])