from sklearn.model_selection import train_test_split

from util.config import load_config


class DatasetSplitter():
    
    def __init__(self, seed, dataset):
        self.seed = seed
        self.dataset = dataset
        
    
    def train_val_test_split(self):
        cf = load_config()
        x = self.dataset['data']
        y = self.dataset['labels']
       
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=cf['train_test_split'], random_state=self.seed, stratify=y
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, train_size=cf['train_val_split'], 
            random_state=self.seed, stratify=y_train
        ) 
        return {
            'train': (x_train, y_train),
            'val': (x_val, y_val),
            'test': (x_test, y_test)
        }