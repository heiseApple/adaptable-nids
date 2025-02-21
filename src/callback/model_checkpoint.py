import os

from callback.callback_lib import Callback


class ModelCheckpoint(Callback):
    """
    Callback for checkpointing: saves the model when the monitored metric improves
    and loads the checkpoint at the beginning of validation and test.
    """
    def __init__(self, monitor, mode):
        super().__init__()
        
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
        self.weights_path = None
        self.checkpoint_filename = 'checkpoint'  # Base name for the checkpoint file

    def on_epoch_end(self, module, epoch):
        """
        At the end of each epoch, checks if the monitored metric has improved
        compared to the best score obtained so far. If yes, saves the model.
        """
        if self.monitor not in module.epoch_outputs:
            raise ValueError(f'Monitor value "{self.monitor}" not found in module outputs.')

        if self.mode not in ['min', 'max']:
            raise ValueError("Mode should be 'min' or 'max'")
        
        current = module.epoch_outputs[self.monitor]
                
        improved = False
        if self.best_score is None:
            # First epoch -> self.best_score is None
            improved = True
        else:
            if self.mode == 'min' and current < self.best_score:
                improved = True
            elif self.mode == 'max' and current > self.best_score:
                improved = True

        if improved:
            if self.weights_path is not None:
                os.remove(self.weights_path) # Remove old checkpoint_ep.pt
            self.best_score = current
            self.weights_path = module.net.save_weights(f'{self.checkpoint_filename}_{epoch+1}')
            
    def _load_weighs(self, module, phase):
        if os.path.exists(self.weights_path):
            print(f"[ModelCheckpoint] Loading checkpoint from {self.weights_path} for {phase}.")
            module.net.load_weights(self.weights_path)
        else:
            raise FileNotFoundError(f'Checkpoint file not found at {self.weights_path}')

    def on_validation_start(self, module):
        """
        At the start of validation, loads the checkpoint if it exists.
        """
        self._load_weighs(module, phase='validation')
        
    def on_test_start(self, module):
        """
        At the start of testing, loads the checkpoint if it exists.
        """
        self._load_weighs(module, phase='testing')