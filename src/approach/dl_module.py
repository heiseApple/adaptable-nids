import torch
import importlib
from argparse import ArgumentParser
from torch import optim

from util.config import load_config
from network.network_factory import build_network

dl_approaches = {
    'scratch' : 'Scratch',
}


class DLModule:
    
    def __init__(self, datamodule, callbacks=None, **kwargs):
        cf = load_config()
        
        gpu = kwargs.get('gpu', cf['gpu'])
        self.device = torch.device('cuda') if gpu and torch.cuda.is_available() else torch.device('cpu')
        print(f'Using device: {self.device}')
        
        self.net = build_network(**kwargs).to(self.device)
        print(self.net)
        
        self.datamodule = datamodule
        self.num_classes = kwargs.get('num_classes', cf['num_classes'])
                
        self.lr = kwargs.get('lr', cf['lr'])
        self.lr_strat = kwargs.get('lr_strat', cf['lr_strat'])
        self.sch_monitor = kwargs.get('sch_monitor', cf['sch_monitor'])
        
        self.max_epochs = kwargs.get('max_epochs', cf['max_epochs'])
        self.min_epochs = kwargs.get('min_epochs', cf['min_epochs'])
        
        self.phase = None
        self.outputs = None
        
        self.callbacks = callbacks if callbacks is not None else []
        
        self.configure_optimizers()
        
        
    @staticmethod
    def get_approach(appr_name, **kwargs):
        Approach = getattr(importlib.import_module(
            name=f'approach.{appr_name}'), dl_approaches[appr_name])
        return Approach(**kwargs)
    
    
    @staticmethod
    def add_appr_specific_args(parent_parser):
        cf = load_config()
        parser = ArgumentParser(
            parents=[parent_parser],
            add_help=True,
            conflict_handler='resolve',
        )
        parser.add_argument('--lr', type=float, default=cf['lr'])
        parser.add_argument('--lr-strat', type=str, default=cf['lr_strat'])
        parser.add_argument('--sch-monitor', type=str, default=cf['sch_monitor'])
        parser.add_argument('--max-epochs', type=int, default=cf['max_epochs'])
        parser.add_argument('--min-epochs', type=int, default=cf['min_epochs'])
        return parser
    
    ####
    # TRAIN, VALIDATION AND TEST
    ####
    
    def fit(self):
        print('='*100)
        self.phase = 'train'

        for cb in self.callbacks: 
            cb.on_fit_start(self)
            
        train_dataloader = self.datamodule.get_train_data()
        val_dataloader = self.datamodule.get_val_data()
        self._fit(train_dataloader, val_dataloader)

        for cb in self.callbacks:
            cb.on_fit_end(self)
        
    def test(self):
        print('='*100)
        self.phase = 'test'
        
        for cb in self.callbacks:
            cb.on_test_start(self)
            
        test_dataloader = self.datamodule.get_test_data()
        self.outputs = self._predict(test_dataloader)
        
        for cb in self.callbacks:
            cb.on_test_end(self)
        
    def validation(self):
        self.phase = 'val'
        
        for cb in self.callbacks:
            cb.on_validation_start(self)
            
        val_dataloader = self.datamodule.get_val_data()
        self.outputs = self._predict(val_dataloader)
        
        for cb in self.callbacks:
            cb.on_validation_end(self)
            
    ####
    # SCHEDULER AND OPTIMIZER
    ####
    
    def configure_optimizers(self):
        cf = load_config()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        
        if self.lr_strat == 'none':
            self.scheduler = None

        elif self.lr_strat == 'lrop':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=cf['lrop_mode'],
                factor=cf['lrop_factor'],
                patience=cf['lrop_patience'],
            )

        elif self.lr_strat == 'cawr':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=cf['cawr_t0'],
                T_mult=cf['cawr_t_mult'],
                eta_min=float(cf['cawr_eta_min'])
            )
            
        else:
            raise ValueError('Scheduler not implemented')
            
    
    def run_scheduler_step(self, monitor_value, epoch=None):
        if self.scheduler is None:
            return
        if self.lr_strat == 'lrop':
            self.scheduler.step(monitor_value)
        elif self.lr_strat == 'cawr':
            self.scheduler.step(epoch)
        else:
            raise ValueError('Scheduler not implemented')
        
        new_lr = self.scheduler.get_last_lr()[0]
        print(f'[Scheduler][Ep{epoch}] Current LR after step: {new_lr:.6f}')