from callback.callback_lib import Callback


class EarlyStoppingCallback(Callback):
    
    def __init__(self, monitor, mode, patience, min_delta, verbose=True):
        super().__init__()
        
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        
        # Inner variables
        self.best_score = None
        self.wait = 0
        self.stopped_epoch = 0
    
    def on_fit_start(self, module):
        # Init inner vars and module.should_stop to False
        self.best_score = None
        self.wait = 0
        self.stopped_epoch = 0
        module.should_stop = False
        
    def on_epoch_end(self, module, epoch):
        if self.monitor not in module.outputs:
            raise ValueError(f'Monitor value "{self.monitor}" not found in module outputs.')
        
        current = module.outputs[self.monitor]
        
        # First epoch, set self.best_score and return
        if self.best_score is None:
            self.best_score = current
            return
        
        improved = False
        
        if self.mode not in ['min', 'max']:
            raise ValueError("Mode should be 'min' or 'max'")
        
        if self.mode == 'min' and (self.best_score - current > self.min_delta):
            improved = True
        elif self.mode == 'max' and (current - self.best_score > self.min_delta):
            improved = True
                    
        if improved:
            # Model has improved, reset self.wait
            self.best_score = current
            self.wait = 0
        else:
            # Model has not improved
            self.wait += 1
        
            # Check patience
            if self.wait >= self.patience:
                # Stop training
                module.should_stop = True
                self.stopped_epoch = epoch + 1                 
                    
    def on_fit_end(self, _):
        if self.stopped_epoch > 0 and self.verbose:
            print(f'[EarlyStopping] Training stopped at epoch {self.stopped_epoch}.' 
                  f' Best val {self.monitor} was {self.best_score:.6f}.')