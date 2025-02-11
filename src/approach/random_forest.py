import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from approach.ml_module import MLModule
from util.config import load_config


class RandomForest(MLModule):
    """
    Wrapper of RandomForestClassifier from sklearn.
    Attributes:
        n_jobs (int): The number of jobs to run in parallel. Default is loaded from configuration.
        verbose (int): The verbosity level. Default is loaded from configuration.
        n_estimators (int): The number of trees in the forest. Default is loaded from configuration.
        max_depth (int): The maximum depth of the tree. Default is loaded from configuration.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cf = load_config()

        self.criterion = kwargs.get('criterion', cf['rf_criterion'])
        self.n_estimators = kwargs.get('n_estimators', cf['rf_n_estimators'])
        self.max_depth = kwargs.get('max_depth', cf['rf_max_depth'])

        self.model = RandomForestClassifier(
            random_state=self.seed,
            criterion=self.criterion,
            n_estimators=self.n_estimators, 
            max_depth=self.max_depth
        )
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        cf = load_config()
        parser = MLModule.add_model_specific_args(parent_parser)
        parser.add_argument('--criterion', type=str, default=cf['rf_criterion'])
        parser.add_argument('--rf-n-estimators', type=int, default=cf['rf_n_estimators'])
        parser.add_argument('--rf-max-depth', type=int, default=cf['rf_max_depth'])
        return parser
    
    def _fit(self, data, labels):
        self.model.fit(data, labels)

    def _predict(self, data, labels):
        probs = self.model.predict_proba(data)
        preds = np.argmax(probs, axis=1)
        
        summary = classification_report(labels, preds, digits=4, output_dict=True, zero_division=0)
        
        return {
            'accuracy': summary['accuracy'],
            'f1_score_macro_avg': summary['macro avg']['f1-score'],
            'labels': labels,
            'preds': preds,
        }
        