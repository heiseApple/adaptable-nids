import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

from approach.ml_module import MLModule
from util.config import load_config


class XGB(MLModule):
    """
    Wrapper of XGBClassifier from xgboost.
    Attributes:
        n_estimators (int): Number of boosting rounds.
        max_depth (int): Maximum tree depth for base learners.
        eval_metric (str): Evaluation metric for validation data.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cf = load_config()

        self.n_estimators = kwargs.get('xgb_n_estimators', cf['xgb_n_estimators'])
        self.max_depth = kwargs.get('xgb_max_depth', cf['xgb_max_depth'])
        self.eval_metric = kwargs.get('xgb_eval_metric', cf['xgb_eval_metric'])
        
        self.classifier = XGBClassifier(
            use_label_encoder=True,
            n_estimators=self.n_estimators, 
            max_depth=self.max_depth,
            eval_metric=self.eval_metric,
        )
        
    @staticmethod
    def add_appr_specific_args(parent_parser):
        cf = load_config()
        parser = MLModule.add_appr_specific_args(parent_parser)
        parser.add_argument('--xgb-n-estimators', type=int, default=cf['xgb_n_estimators'])
        parser.add_argument('--xgb-max-depth', type=int, default=cf['xgb_max_depth'])
        parser.add_argument('--xgb-eval-metric', type=str, default=cf['xgb_eval_metric'])
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