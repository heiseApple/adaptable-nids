import numpy as np
from sklearn.model_selection import train_test_split

from data.util import DataSplits
from data.data_reader import get_data_labels
from util.config import load_config


class DataManager:
    """
    Manages and prepares data for training, validation, and testing.
    """
    def __init__(self, args):
        self.args = args
        self.seed = self.args.seed
        
    def get_dataset_splits(self):
        # Loads source and target datasets based on provided arguments.
        dataset_args = {
            'num_pkts': self.args.num_pkts,
            'fields': self.args.fields,
            'is_flat': self.args.is_flat,
            'seed': self.args.seed,
        }
        
        # Splits the source dataset and, if present, also the target dataset.
        # It also sets the number of classes and checks for compatibility.
        src_dataset = get_data_labels(dataset=self.args.src_dataset, **dataset_args)
        src_splits, num_classes = self._train_val_test_split(dataset=src_dataset)
        
        if self.args.trg_dataset is not None:
            trg_dataset = get_data_labels(dataset=self.args.trg_dataset, **dataset_args)
            trg_splits, trg_num_classes = self._train_val_test_split(dataset=trg_dataset)
            
            assert num_classes == trg_num_classes, (
                'Mismatch between the classes of the source and target datasets'
            )
            
            # For a few-shot setup (k samples per trg class)
            if self.args.k is not None:
                x_sampled, y_sampled = self._sample_k_per_class(trg_splits['train'], k=self.args.k)
                trg_splits['train'] = (x_sampled, y_sampled) 
                trg_splits['val'] = (x_sampled, y_sampled) # Same data used for training
        else:
            trg_splits = None
            
        return src_splits, trg_splits, num_classes
    
    
    def _train_val_test_split(self, dataset):
        """
        Splits the dataset into training, validation, and test sets.
        """
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
        return DataSplits(
            train=(x_train, y_train),
            val=(x_val, y_val),
            test=(x_test, y_test)
        ), len(np.unique(y))
        
        
    def _sample_k_per_class(self, data, k):
        x, y = data
        rng = np.random.default_rng(self.args.k_seed)
        
        # Retrieve unique classes and their counts
        classes, counts = np.unique(y, return_counts=True)

        # Ensure every class has at least k samples
        if np.any(counts < k):
            raise ValueError(f'Not all classes have at least {k} samples')
        
        # Generate a random permutation of all sample indices
        perm = rng.permutation(len(y))
        # Reorder labels according to the permutation
        y_perm = y[perm]
        
        # Map each label in y_perm to its index in the sorted array of unique classes
        idx_in_classes = np.searchsorted(classes, y_perm)
        
        # Prepare a one-hot matrix for each sample vs. each class
        num_classes = len(classes)
        one_hot = np.zeros((len(y), num_classes), dtype=int)
        one_hot[np.arange(len(y)), idx_in_classes] = 1
        
        # Compute the cumulative sum column-wise
        # one_hot_csum[i, c] = number of samples of class c up to (and including) index i
        one_hot_csum = one_hot.cumsum(axis=0)
        
        # Determine the rank of each sample within its class
        # (i.e., 1st sample of that class, 2nd, etc.)
        rank_of_sample = one_hot_csum[np.arange(len(y)), idx_in_classes]
        
        # Keep only the first k samples for each class
        mask = rank_of_sample <= k
        
        # Retrieve original indices from the permuted list
        final_indices = perm[mask]
        
        # Sort
        sort_order = np.argsort(y[final_indices])
        final_indices = final_indices[sort_order]

        # Return the sampled data and labels
        return x[final_indices], y[final_indices]
    