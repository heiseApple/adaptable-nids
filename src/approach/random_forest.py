import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from approach.ml_module import MLModule
from util.config import load_config


class RandomForest(MLModule):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cf = load_config()

        self.n_jobs = kwargs.get('n_jobs', cf['rf_n_jobs'])
        self.verbose = kwargs.get('verbose', cf['rf_verbose'])
        self.n_estimators = kwargs.get('n_estimators', cf['rf_n_estimators'])
        self.max_depth = kwargs.get('max_depth', cf['rf_max_depth'])
        
        self.model = RandomForestClassifier(
            random_state=self.seed,
            n_jobs=self.n_jobs, 
            verbose=self.verbose, 
            n_estimators=self.n_estimators, 
            max_depth=self.max_depth
        )
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        cf = load_config()
        parser = MLModule.add_model_specific_args(parent_parser)
        parser.add_argument('--n-jobs', type=int, default=cf['rf_n_jobs'])
        parser.add_argument('--verbose', type=int, default=cf['rf_verbose'])
        parser.add_argument('--n-estimators', type=int, default=cf['rf_n_estimators'])
        parser.add_argument('--max-depth', type=int, default=cf['rf_max_depth'])
        return parser
    
    def _fit(self, data, labels):
        self.model.fit(data, labels)

    def _predict(self, data, labels):
        prob = self.model.predict_proba(data)
        pred = np.argmax(prob, axis=1)
        
        summary = classification_report(labels, pred, digits=4, output_dict=True, zero_division=0)
        print(summary)
        