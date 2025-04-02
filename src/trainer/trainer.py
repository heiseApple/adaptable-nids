import json

from approach.approach_factory import get_approach
from data.util import DataSplits, concat_dataset
from util.directory_manager import DirectoryManager


class Trainer:
    """
    The Trainer class is responsible for managing the training and evaluation process.
    It initializes with the given arguments and datasets, and provides methods to run
    the training, validation, and testing phases using the specified approach.
    """
    def __init__(self, args, dataset_splits):
        self.args = args
        self.dict_args = vars(args)
        self.dataset_splits = dataset_splits
        self.dm = DirectoryManager()
        
        
    def run(self):
        # Get dataset splits
        src_splits, trg_splits, self.dict_args['num_classes'] = self.dataset_splits
                        
        self._save_dict_args()

        # Init approach
        approach = get_approach(self.args.approach, **self.dict_args)
        print('='*100)
        
        # Handle different training scenarios
        if self.args.trg_dataset is None:
            self._handle_sup_src(approach, src_splits)
        elif self.args.is_appr_unsup:
            self._handle_sup_src_unsup_trg(approach, src_splits, trg_splits)
        else:
            self._handle_sup_src_sup_trg(approach, src_splits, trg_splits)
            
    
    def _handle_sup_src(self, approach, src_splits):
        # Train, validation, and test only on src_dataset
        approach.datamodule = src_splits.get_datamodule(**self.dict_args)
        self._fit_val(approach)
        self._test(approach)
        
        
    def _handle_sup_src_unsup_trg(self, approach, src_splits, trg_splits):
        """
        Handle unsupervised domain adaptation scenario.
        """
        if self.args.n_tasks == 1:
            # n_task == 1 -> monolithic training (only for ML semi-sup appoaches)
            # Combine src_dataset and trg_dataset train/val splits (trg train is unsupervised)
            combined_train = concat_dataset(src_splits.train, trg_splits.train, unsup='d2')
            combined_val = concat_dataset(src_splits.val, trg_splits.val)
            approach.datamodule = DataSplits(
                train=combined_train,
                val=combined_val,
                test=None
            ).get_datamodule(**self.dict_args)
            self._fit_val(approach)
            self._cross_dataset_test(approach, src_splits, trg_splits)
        else:
            # n_task == 2 -> sequential training
            if not self.args.skip_t1:
                # Train and val on src
                src_datamodule = src_splits.get_datamodule(**self.dict_args)
                approach.datamodule = src_datamodule # Supervised data from src
                print(f'[Trainer] Starting training on source dataset: {self.args.src_dataset}')
                self._fit_val(approach)

            self.dm.change_log_dir(task='trg') # Switch log_dir to trg 
            
            # Target adaptation
            approach = self._reset_approach(weights_path=self.args.weights_path)
            approach.task = 'trg'
            approach.datamodule = trg_splits.get_datamodule(**self.dict_args) # Unsupervised data from trg
            print(f'[Trainer] Starting training on target dataset: {self.args.trg_dataset}')
            self._adapt_val(approach, train_dataloader=src_datamodule.get_train_data())
            self._cross_dataset_test(approach, src_splits, trg_splits) 
            
            
    def _handle_sup_src_sup_trg(self, approach, src_splits, trg_splits):
        """
        Handle monolithic and sequential training on source and target.
        """
        if self.args.n_tasks == 1:
            # n_task == 1 -> monolithic training
            # Combine src_dataset and trg_dataset train/val splits
            combined_train = concat_dataset(src_splits.train, trg_splits.train)
            combined_val = concat_dataset(src_splits.val, trg_splits.val)
            approach.datamodule = DataSplits(
                train=combined_train,
                val=combined_val,
                test=None
            ).get_datamodule(**self.dict_args)
            self._fit_val(approach)
            self._cross_dataset_test(approach, src_splits, trg_splits)
        else:
            # n_task == 2 -> sequential training
            if not self.args.skip_t1:
                # Train and val on src
                approach.datamodule = src_splits.get_datamodule(**self.dict_args)
                print(f'[Trainer] Starting training on source dataset: {self.args.src_dataset}')
                self._fit_val(approach)
            
            self.dm.change_log_dir(task='trg') # Switch the log_dir to trg 
            
            # Target adaptation
            approach = self._reset_approach(weights_path=self.args.weights_path)
            approach.task = 'trg'
            approach.datamodule = trg_splits.get_datamodule(**self.dict_args)
            print(f'[Trainer] Starting training on target dataset: {self.args.trg_dataset}')
            self._adapt_val(approach)
            self._cross_dataset_test(approach, src_splits, trg_splits)
        
        
    def _cross_dataset_test(self, approach, src_splits, trg_splits):
        # Test on target
        approach.task = 'trg'
        print(f'[Trainer] Starting test on target dataset: {self.args.trg_dataset}')
        self.dm.change_log_dir(task='trg')
        self._test(approach, test_dataset=trg_splits.test)

        # Test on source
        print(f'[Trainer] Starting test on source dataset: {self.args.src_dataset}')
        approach.task = 'src'
        self.dm.change_log_dir(task='src')
        self._test(approach, test_dataset=src_splits.test)
         
    def _fit_val(self, approach):
        approach.fit()
        approach.validate()
        
    def _adapt_val(self, approach, **kwargs):
        approach.adapt(**kwargs)
        approach.validate()
        
    def _test(self, approach, test_dataset=None):
        if test_dataset is not None:
            approach.datamodule.set_test_dataset(test_dataset)     
        approach.test()
     
    def _reset_approach(self, weights_path=None):        
        approach = get_approach(
            approach_name=self.args.approach, 
            fs_task=self.args.k is not None, 
            **self.dict_args
        )
        # Loads weights from the first task or from weights_path
        approach.net.load_weights(weights_path or self.dm.checkpoint_path) 
        print(f'[Trainer] Loaded network weights from {weights_path or self.dm.checkpoint_path}')
        return approach
            
    def _save_dict_args(self):
        with open(f'{self.dm.log_dir}/dict_args.json', 'w') as f:
            json.dump(self.dict_args, f)
    