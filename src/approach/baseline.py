from torch import nn

from approach.dl_module import DLModule
from callback.freeze_backbone import FreezeBackbone
from util.config import load_config


class Baseline(DLModule):
    """
    Baseline class for a deep learning module that includes training, validation, 
    and adaptation strategies (finetuning or freezing).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cf = load_config()
        
        self.criterion = nn.CrossEntropyLoss()
        self.adaptation_strat = kwargs.get('adaptation_strat', cf['adaptation_strat'])
        self.callbacks.append(FreezeBackbone())
        
        
    @staticmethod
    def add_appr_specific_args(parent_parser):
        cf = load_config()
        parser = DLModule.add_appr_specific_args(parent_parser)
        parser.add_argument('--adaptation-strat', type=str, default=cf['adaptation_strat'], 
                            choices=['finetuning', 'freezing'])
        return parser
    
    
    def _fit_step(self, batch_x, batch_y):
        logits = self.net(batch_x)
        loss = self.criterion(logits, batch_y)
        return loss, logits
                
            
    def _predict_step(self, batch_x, batch_y):
        logits = self.net(batch_x)
        loss = self.criterion(logits, batch_y)
        return loss, logits
    
        
    def _adapt(self, train_dataloader, val_dataloader):
        self._fit(train_dataloader, val_dataloader)