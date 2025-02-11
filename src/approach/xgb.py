import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

from approach.ml_module import MLModule
from util.config import load_config


class XGB(MLModule):
    
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
        prob = self.model.predict_proba(data)
        pred = np.argmax(prob, axis=1)
        
        summary = classification_report(labels, pred, digits=4, output_dict=True, zero_division=0)
        print(summary)