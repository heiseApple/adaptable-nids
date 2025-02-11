import json
import importlib
import numpy as np
from argparse import ArgumentParser

from util.config import load_config
from util.directory_manager import DirectoryManager

ml_approaches = {
    'random_forest' : 'RandomForest',
    'xgb' : 'XGB'
}

class MLModule():
    
    def __init__(self, **kwargs):
        cf = load_config()

        self.seed = kwargs.get('seed', cf['seed'])
        self.phase = None
        
    @staticmethod
    def get_approach(ml_name, **kwargs):
        # Dynamically import and return the specified ML approach class
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
        # Set phase to 'train' and fit the model
        self.phase = 'train'
        data, labels = train_dataset
        self._fit(data, labels)
        
    def predict(self, test_dataset):
        # Set phase to 'test' and test the model
        self.phase = 'test'
        data, labels = test_dataset
        self.outputs = self._predict(data, labels)
        self.on_predict_end()
        
    def validate(self, val_dataset):
        # Set phase to 'val' and validate the model
        self.phase = 'val'
        data, labels = val_dataset
        self.outputs = self._predict(data, labels)
        self.on_predict_end()
        
    def on_predict_end(self):
        # Save prediction results and metrics
        dm = DirectoryManager()
        path = dm.mkdir(f'{self.phase}')
        np.savez_compressed(f'{path}/labels.npz', labels=self.outputs['labels'])
        np.savez_compressed(f'{path}/preds.npz', preds=self.outputs['preds'])
        
        res = {
            'accuracy': self.outputs['accuracy'],
            'f1_score_macro_avg': self.outputs['f1_score_macro_avg'],
        }
        with open(f'{dm.log_dir}/{self.phase}_results.json', 'w') as f:
            json.dump(res, f)