import importlib
from argparse import ArgumentParser

from util.config import load_config

ml_approaches = {
    'random_forest' : 'RandomForest',
    'xgb' : 'XGB'
}


class MLModule():
    
    def __init__(self, **kwargs):
        cf = load_config()
        self.seed = kwargs.get('seed', cf['seed'])
        
    @staticmethod
    def get_approach(ml_name, **kwargs):
        Approach = getattr(importlib.import_module(
            name=f'approach.{ml_name}'), ml_approaches[ml_name])
        return Approach(**kwargs)
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(
            parents=[parent_parser],
            add_help=True,
            conflict_handler='resolve',
        )
        return parser
    
    def fit(self, train_dataset):
        data, labels = train_dataset
        self._fit(data, labels)
        
    def predict(self, test_dataset):
        data, labels = test_dataset
        self._predict(data, labels)