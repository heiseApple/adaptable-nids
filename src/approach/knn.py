import numpy as np
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

from approach.ml_module import MLModule
from util.config import load_config


class KNN(MLModule):
    """
    Wrapper of KNeighborsClassifier from sklearn.
    Attributes:
        n_neighbors (int): The number of neighbors to use. Default is loaded from configuration.
        weights (str): The weight function used in prediction. Default is loaded from configuration.
        p (int): The power parameter for the Minkowski metric. Default is loaded from configuration.
        metric (str): The distance metric to use. Default is loaded from configuration.
        
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cf = load_config()

        self.n_neighbors = kwargs.get('knn_n_neighbors', cf['knn_n_neighbors'])
        self.weights = kwargs.get('knn_weights', cf['knn_weights'])
        self.p = kwargs.get('knn_p', cf['knn_p'])
        self.metric = kwargs.get('knn_metric', cf['knn_metric'])
        
        self.classifier = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            p=self.p,
            metric=self.metric,
        )
        
    @staticmethod
    def add_appr_specific_args(parent_parser):
        cf = load_config()
        parser = MLModule.add_appr_specific_args(parent_parser)
        parser.add_argument('--knn-n-neighbors', type=int, default=cf['knn_n_neighbors'])
        parser.add_argument('--knn-weights', type=str, default=cf['knn_weights'])
        parser.add_argument('--knn-p', type=int, default=cf['knn_p'])
        parser.add_argument('--knn-metric', type=str, default=cf['knn_metric'])
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
        