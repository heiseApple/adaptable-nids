import importlib
from argparse import ArgumentParser
from torch import optim

from util.config import load_config

dl_approaches = {
    'scratch' : 'Scratch',
}


class MLModule():
    
    def __init__(self, **kwargs):
        cf = load_config()
        
        self.lr = kwargs.get('lr', cf['lr'])
        self.lr_strat = kwargs.get('lr_strat', cf['lr_strat'])
        self.max_epochs = kwargs.get('max_epochs', cf['max_epochs'])
        self.min_epochs = kwargs.get('min_epochs', cf['min_epochs'])
        self.phase = None
        
        
    @staticmethod
    def get_approach(dl_name, **kwargs):
        Approach = getattr(importlib.import_module(
            name=f'approach.{dl_name}'), dl_approaches[dl_name])
        return Approach(**kwargs)
    
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        cf = load_config()
        parser = ArgumentParser(
            parents=[parent_parser],
            add_help=True,
            conflict_handler='resolve',
        )
        parser.add_argument('--lr', type=float, default=cf['lr'])
        parser.add_argument('--lr-strat', type=str, default=cf['lr_strat'])
        parser.add_argument('--max-epochs', type=int, default=cf['max_epochs'])
        parser.add_argument('--min-epochs', type=int, default=cf['min_epochs'])
        return parser
    
    def fit(self, train_dataset):
        # Set phase to 'train' and fit the model
        self.phase = 'train'
        pass
        
    def predict(self, test_dataset):
        # Set phase to 'test' and test the model
        self.phase = 'test'
        pass
        
    def validate(self, val_dataset):
        # Set phase to 'test' and test the model
        self.phase = 'test'
        pass
    
    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        if self.lr_strat == 'none':
            self.scheduler = None

        elif self.lr_strat == 'lrop':
            self.scheduler = optim.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.1,
                patience=5,
                verbose=True
            )

        elif self.lr_strat == 'cawr':
            self.scheduler = optim.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=1,
                eta_min=0
            )
        