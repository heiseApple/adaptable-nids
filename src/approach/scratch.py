import torch
from torch import nn
from tqdm import tqdm
from torchmetrics import Accuracy, F1Score

from approach.dl_module import DLModule


class Scratch(DLModule):
    """
    TODO
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.criterion = nn.CrossEntropyLoss()
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = DLModule.add_model_specific_args(parent_parser)
        return parser
    
    
    def _fit(self, train_dataloader, val_dataloader):
        
        accuracy = Accuracy(num_classes=self.num_classes, task='multiclass')
        f1_score = F1Score(num_classes=self.num_classes, average='macro', task='multiclass')
        
        val_score = None
        
        # Training loop
        for epoch in range(self.max_epochs):
            self.net.train()
            
            accuracy.reset()
            f1_score.reset()
            
            running_loss = 0.0
            
            desc = f'Ep[{epoch+1}/{self.max_epochs}]'
            if val_score is not None:
                desc = (f'{desc}||trn loss:{epoch_loss:.4f}||trn acc:{epoch_acc:.4f}'
                        f'||trn f1:{epoch_f1:.4f}||val {self.sch_monitor}:{val_score:.4f}')
            train_loop = tqdm(train_dataloader, desc=desc, leave=False)

            for batch_x, batch_y in train_loop:                 
                # Forward pass
                logits = self.net(batch_x)
                loss = self.criterion(logits, batch_y.long())
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                
                # Metrics
                preds = torch.argmax(logits, dim=1)
                accuracy.update(preds, batch_y)
                f1_score.update(preds, batch_y)
                          
            epoch_loss = running_loss / len(train_dataloader)
            epoch_acc = accuracy.compute().item()
            epoch_f1  = f1_score.compute().item()
            
            val_score = self._predict(val_dataloader, on_train_epoch_end=True)[self.sch_monitor]
            self.run_scheduler_step(monitor_value=val_score, epoch=epoch+1)
            
            
    def _predict(self, dataloader, on_train_epoch_end=False):
        self.net.eval()
        
        accuracy = Accuracy(num_classes=self.num_classes, task='multiclass')
        f1_score = F1Score(num_classes=self.num_classes, average='macro', task='multiclass')
        
        all_labels, all_preds, all_logits = [], [], []
        running_loss = 0.0
        
        desc = '[val]' if on_train_epoch_end else f'[{self.phase}]'
        with torch.no_grad():
            
            for batch_x, batch_y in tqdm(dataloader, desc=desc, leave=not self.phase=='train'):
                logits = self.net(batch_x)
                preds = torch.argmax(logits, dim=1)
                
                loss = self.criterion(logits, batch_y.long())
                running_loss += loss.item() * batch_y.shape[0]
                
                # Metrics
                accuracy.update(preds, batch_y)
                f1_score.update(preds, batch_y)
                                
                all_labels.append(batch_y)
                all_preds.append(preds)
                all_logits.append(logits)
                
        labels = torch.cat(all_labels, dim=0)
        preds = torch.cat(all_preds, dim=0)
        logits = torch.cat(all_logits, dim=0)
        
        eval_loss = running_loss / labels.shape[0] if labels.shape[0] > 0 else 0.0
        
        return {
            'accuracy' : accuracy.compute().item(),
            'f1_score_macro' : f1_score.compute().item(),
            'loss' : eval_loss,
            'labels': labels.detach().cpu().numpy(),
            'preds': preds.detach().cpu().numpy(),
            'logits': logits.detach().cpu().numpy(),
        }