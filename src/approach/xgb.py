import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

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

        self.n_estimators = kwargs.get('n_estimators', cf['xgb_n_estimators'])
        self.max_depth = kwargs.get('max_depth', cf['xgb_max_depth'])
        self.eval_metric = kwargs.get('eval_metric', cf['xgb_eval_metric'])
        
        self.model = XGBClassifier(
            use_label_encoder=True,
            n_estimators=self.n_estimators, 
            max_depth=self.max_depth,
            eval_metric=self.eval_metric
        )
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        cf = load_config()
        parser = MLModule.add_model_specific_args(parent_parser)
        parser.add_argument('--n-estimators', type=int, default=cf['xgb_n_estimators'])
        parser.add_argument('--max-depth', type=int, default=cf['xgb_max_depth'])
        parser.add_argument('--eval-metric', type=str, default=cf['xgb_eval_metric'])
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