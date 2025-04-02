import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

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

        self.criterion = kwargs.get('rf_criterion', cf['rf_criterion'])
        self.n_estimators = kwargs.get('rf_n_estimators', cf['rf_n_estimators'])
        self.max_depth = kwargs.get('rf_max_depth', cf['rf_max_depth'])

        self.classifier = RandomForestClassifier(
            random_state=self.seed,
            criterion=self.criterion,
            n_estimators=self.n_estimators, 
            max_depth=self.max_depth,
            verbose=True
        )
        
    @staticmethod
    def add_appr_specific_args(parent_parser):
        cf = load_config()
        parser = MLModule.add_appr_specific_args(parent_parser)
        parser.add_argument('--rf-criterion', type=str, default=cf['rf_criterion'])
        parser.add_argument('--rf-n-estimators', type=int, default=cf['rf_n_estimators'])
        parser.add_argument('--rf-max-depth', type=int, default=cf['rf_max_depth'])
        return parser
    
    def _fit(self, data, labels):
        self.classifier.fit(data, labels)

    def _predict(self, data, labels):
        probs = self.classifier.predict_proba(data)
        preds = np.argmax(probs, axis=1)
        
        accuracy = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'f1_score_macro': f1_macro,
            'labels': labels,
            'preds': preds,
        }
        