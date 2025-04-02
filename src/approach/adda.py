import sys
from torch import nn
from tqdm import tqdm
from copy import deepcopy

from approach.dl_module import DLModule
from data.util import InfiniteDataIterator
from module.domain_discriminator import DomainDiscriminator
from module.gradient_reverse_function import WarmStartGradientReverseLayer
from module.loss import DomainAdversarialLoss
from util.config import load_config

disable_tqdm = not sys.stdout.isatty()


class ADDA(DLModule):
    """
    [[Link to Source Code]](https://github.com/thuml/Transfer-Learning-Library/blob/master/examples/domain_adaptation/image_classification/adda.py)
    ADDA is a class that implements the ADDA (Adversarial Discriminative Domain Adaptation) algorithm for domain adaptation,
    as described in "Adversarial Discriminative Domain Adaptation".
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cf = load_config()
                
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        self.discr_hidden_size = kwargs.get('discr_hidden_size', cf['discr_hidden_size'])
        self.wsgrl_alpha = kwargs.get('wsgrl_alpha', cf['wsgrl_alpha'])
        self.wsgrl_lo = kwargs.get('wsgrl_lo', cf['wsgrl_lo'])
        self.wsgrl_hi = kwargs.get('wsgrl_hi', cf['wsgrl_hi'])
        self.wsgrl_max_iters = kwargs.get('wsgrl_max_iters', cf['wsgrl_max_iters'])
        self.wsgrl_auto_step = kwargs.get('wsgrl_auto_step', cf['wsgrl_auto_step'])
        self.adapt_lr = kwargs.get('adapt_lr', cf['adapt_lr'])
        self.adapt_epochs = kwargs.get('adapt_epochs', cf['adapt_epochs'])
        self.iter_per_epoch = kwargs.get('iter_per_epoch', cf['iter_per_epoch'])
        
        
    @staticmethod
    def add_appr_specific_args(parent_parser):
        cf = load_config()
        parser = DLModule.add_appr_specific_args(parent_parser)
        parser.add_argument('--discr-hidden-size', type=int, default=cf['discr_hidden_size'])
        parser.add_argument('--wsgrl-alpha', type=float, default=cf['wsgrl_alpha'])
        parser.add_argument('--wsgrl-lo', type=float, default=cf['wsgrl_lo'])
        parser.add_argument('--wsgrl-hi', type=float, default=cf['wsgrl_hi'])
        parser.add_argument('--wsgrl-max-iters', type=int, default=cf['wsgrl_max_iters'])
        parser.add_argument('--wsgrl-auto-step', action='store_true', default=cf['wsgrl_auto_step'])
        parser.add_argument('--adapt-lr', type=float, default=cf['adapt_lr'])
        parser.add_argument('--adapt-epochs', type=int, default=cf['adapt_epochs'])
        parser.add_argument('--iter-per-epoch', type=int, default=cf['iter_per_epoch'])
        return parser
    
    
    def _fit_step(self, batch_x, batch_y):
        logits = self.net(batch_x)
        loss = self.ce_loss(logits, batch_y)
        return loss, logits
    
    
    def _predict_step(self, batch_x, batch_y):
        logits = self.net(batch_x)
        loss = self.ce_loss(logits, batch_y)
        return loss, logits
    
    def _adapt(self, adapt_dataloader, val_dataloader, train_dataloader):
        # Infinite iterators
        src_dataloader = InfiniteDataIterator(train_dataloader, device=self.device) 
        trg_dataloader = InfiniteDataIterator(adapt_dataloader, device=self.device) 
        
        # Source network is completely frozen
        source_net = deepcopy(self.net)
        source_net.freeze_net()
        source_net.freeze_bn()
        
        domain_discriminator = DomainDiscriminator(
            in_feature=self.net.out_features_size,
            hidden_size=self.discr_hidden_size,
        )
        
        # Domain adaptation loss function
        wsgrl = WarmStartGradientReverseLayer(
            alpha=self.wsgrl_alpha, lo=self.wsgrl_lo, hi=self.wsgrl_hi, 
            max_iters=self.wsgrl_max_iters, auto_step=self.wsgrl_auto_step
        )
        da_loss = DomainAdversarialLoss(domain_discriminator, grl=wsgrl).to(self.device)
        
        self.lr = self.adapt_lr
        params = list(self.net.backbone.parameters()) + list(domain_discriminator.parameters())
        self.configure_optimizers(params=params)
        
        # Adaptation loop       
        for epoch in range(self.adapt_epochs):
            self.net.train()
            da_loss.train()
            
            for cb in self.callbacks:
                cb.on_epoch_start(self, epoch)
                
            running_loss = 0.0
            
            postfix = {
                'DA loss': f'{self.epoch_outputs["train_loss"]:.4f}', 
                'discr acc':  f'{self.epoch_outputs["train_accuracy"]:.4f}',
                f'val {self.sch_monitor}': f'{val_score:.4f}'
            } if epoch > 0 else {} 
            adapt_loop = tqdm(
                range(self.iter_per_epoch), desc=f'Ep[{epoch+1}/{self.adapt_epochs}]',  
                postfix=postfix, leave=False, disable=disable_tqdm
            )
            for i in adapt_loop:
                # Get batches
                batch_x_s, _ = next(src_dataloader)
                batch_x_t, _ = next(trg_dataloader)
                batch_x_s, batch_x_t = batch_x_s.to(self.device), batch_x_t.to(self.device)
                
                # Embedding and domain adaptation loss
                _, batch_emb_s = source_net(batch_x_s, return_feat=True)
                _, batch_emb_t = self.net(batch_x_t, return_feat=True)
                
                loss = da_loss(f_s=batch_emb_s, f_t=batch_emb_t)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                
            # Validation on adapt epoch end
            self.epoch_outputs = self._predict(val_dataloader, on_train_epoch_end=True)
            val_score = self.epoch_outputs[self.sch_monitor]
            
            self.epoch_outputs['train_loss'] = running_loss / self.iter_per_epoch
            self.epoch_outputs['train_accuracy'] = da_loss.domain_discriminator_accuracy
            
            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch)
                
            if self.should_stop:
                break  # Early stopping
            
            self.run_scheduler_step(monitor_value=val_score, epoch=epoch + 1)

        