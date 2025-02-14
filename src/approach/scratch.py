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
    def add_appr_specific_args(parent_parser):
        parser = DLModule.add_appr_specific_args(parent_parser)
        return parser
    
    
    def _fit(self, train_dataloader, val_dataloader):
        
        accuracy = Accuracy(num_classes=self.num_classes, task='multiclass').to(self.device)
        f1_score = F1Score(num_classes=self.num_classes, average='macro', task='multiclass').to(self.device)

        for epoch in range(self.max_epochs):
            self.net.train()

            for cb in self.callbacks:
                cb.on_epoch_start(self, epoch)

            running_loss = 0.0
            all_labels, all_preds = [], []

            postfix = {
                'trn loss': f'{epoch_loss:.4f}', 'trn acc': f'{epoch_acc:.4f}',
                'trn f1': f'{epoch_f1:.4f}', f'val {self.sch_monitor}': f'{val_score:.4f}'
            } if epoch>0 else {}
            train_loop = tqdm(
                train_dataloader, desc=f'Ep[{epoch+1}/{self.max_epochs}]',  
                postfix=postfix, leave=False
            )
            
            
            for batch_x, batch_y in train_loop:
                # Move data on self.device
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
                # Forward pass e Loss
                logits = self.net(batch_x)
                loss = self.criterion(logits, batch_y.long())

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                # Metrics
                preds = torch.argmax(logits, dim=1)
                all_labels.append(batch_y)
                all_preds.append(preds)
                
            epoch_loss = running_loss / len(train_dataloader)

            epoch_acc = accuracy(torch.cat(all_preds), torch.cat(all_labels)).item()
            epoch_f1 = f1_score(torch.cat(all_preds), torch.cat(all_labels)).item()

            # Validation on fit epoch end
            self.epoch_outputs = self._predict(val_dataloader, on_train_epoch_end=True)
            val_score = self.epoch_outputs[self.sch_monitor]

            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch)

            if self.should_stop:
                break  # Early stopping

            self.run_scheduler_step(monitor_value=val_score, epoch=epoch + 1)
                
            
    def _predict(self, dataloader, on_train_epoch_end=False):
        self.net.eval()
        
        accuracy = Accuracy(num_classes=self.num_classes, task='multiclass').to(self.device)
        f1_score = F1Score(num_classes=self.num_classes, average='macro', task='multiclass').to(self.device)
        
        all_labels, all_preds, all_logits = [], [], []
        running_loss = 0.0
        
        desc = '[val]' if on_train_epoch_end else f'[{self.phase}]'
        with torch.no_grad():
            
            for batch_x, batch_y in tqdm(dataloader, desc=desc, leave=not self.phase=='train'):
                # Move data on self.device
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                logits = self.net(batch_x)
                preds = torch.argmax(logits, dim=1)
                
                loss = self.criterion(logits, batch_y.long())
                running_loss += loss.item() * batch_y.shape[0]
                
                all_labels.append(batch_y)
                all_preds.append(preds)
                all_logits.append(logits)
                
        labels = torch.cat(all_labels, dim=0)
        preds = torch.cat(all_preds, dim=0)
        logits = torch.cat(all_logits, dim=0)
        
        eval_loss = running_loss / labels.shape[0] if labels.shape[0] > 0 else 0.0
        
        return {
            'accuracy' : accuracy(torch.cat(all_preds), torch.cat(all_labels)).item(),
            'f1_score_macro' : f1_score(torch.cat(all_preds), torch.cat(all_labels)).item(),
            'loss' : eval_loss,
            'labels': labels.detach().cpu().numpy(),
            'preds': preds.detach().cpu().numpy(),
            'logits': logits.detach().cpu().numpy(),
        }