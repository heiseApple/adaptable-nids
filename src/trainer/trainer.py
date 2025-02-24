import json

from data.data_manager import DataManager
from approach.approach_factory import get_approach
from util.directory_manager import DirectoryManager


class Trainer:
    """
    The Trainer class is responsible for managing the training and evaluation process.
    It initializes with the given arguments and datasets, and provides methods to run
    the training, validation, and testing phases using the specified approach.
    """
    def __init__(self, args, data_manager: DataManager):
        self.args = args
        self.dict_args = vars(args)
        self.data_manager = data_manager
            
        
    def run(self): 
        """
        Main method that:
         1. Performs dataset splitting.
         2. Initializes the desired approach.
         3. Manages the training/validation/test flow based on n_tasks and appr_type.
        """
        # Get dataset splits
        src_splits, trg_splits, self.dict_args['num_classes'] = self.data_manager.split_dataset()
        
        self._save_dict_args()
        
        # Init approach
        approach = get_approach(approach_name=self.args.approach, **self.dict_args)
        
        # Only src dataset is available
        if self.args.trg_dataset is None:
            # Train, validation, and test only on src_dataset
            approach.datamodule = self.data_manager.get_datamodule(**src_splits) 
            self._fit_evaluate(approach)
            self._test(approach)
            return
        
        if self.args.n_tasks == 1:
            # n_task == 1 -> monolithic training
            # Combine src_dataset and trg_dataset train splits
            combined_train = self.data_manager.concat_dataset(
                src_splits['train'], trg_splits['train']
            )
            # Val on src
            approach.datamodule = self.data_manager.get_datamodule(
                train=combined_train,
                val=src_splits['val'],
                test=src_splits['test']
            )
            self._fit_evaluate(approach)
            # Test on both
            self._test(approach) # On src
            self._test(approach, trg_test_dataset=trg_splits['test']) # On trg
            
        else:
            # n_task == 2 -> sequential training
            if self.args.appr_type == 'ml':
                raise ValueError('ML approaches do not support multiple tasks')
            
            if not self.args.skip_t1:
                # Train, val, and test on src
                print(f'[Trainer] Starting task on source dataset: {self.args.src_dataset}')
                approach.datamodule = self.data_manager.get_datamodule(**src_splits)
                self._fit_evaluate(approach)
                self._test(approach)
            
            approach = self._reset_approach(weights_path=self.args.weights_path)
            
            # Train, val, and test on trg
            print(f'[Trainer] Starting task on target dataset: {self.args.trg_dataset}')
            approach.datamodule = self.data_manager.get_datamodule(**trg_splits)
            self._adapt_evaluate(approach)
            self._test(approach)
            
    
    def _fit_evaluate(self, approach):
        approach.fit()
        approach.validate()
        
    def _adapt_evaluate(self, approach):
        approach.adapt()
        approach.validate()
        
    def _test(self, approach, trg_test_dataset=None):
        if trg_test_dataset is not None:
            print(f'[Trainer] Starting test on target dataset: {self.args.trg_dataset}')
            DirectoryManager().update_log_dir() # Switch the log_dir from src to trg 
            approach.datamodule.set_test_dataset(trg_test_dataset)
            
        approach.test()
            
            
    def _reset_approach(self, weights_path=None):
        dm = DirectoryManager() 
        dm.update_log_dir() # Switch the log_dir from src to trg 
        approach = get_approach(
            approach_name=self.args.approach, 
            fs_task=self.args.k is not None, 
            **self.dict_args
        )
        # Loads weights from the first task or from weights_path
        approach.net.load_weights(weights_path or dm.checkpoint_path) 
        print(f'[Trainer] Loaded network weights from {weights_path or dm.checkpoint_path}')
        return approach
            
    def _save_dict_args(self):
        dm = DirectoryManager(self.args.log_dir)
        with open(f'{dm.log_dir}/dict_args.json', 'w') as f:
            json.dump(self.dict_args, f)
    