import os
import pandas as pd

from callback.callback_lib import Callback
from util.directory_manager import DirectoryManager


class SaveTrainLog(Callback):
    """
    Callback that, at the end of each epoch, collects the training and validation
    postfix metrics and appends them as a new row to a file on disk.
    
    The file can be in CSV or Parquet format.
    The file is stored in the "epoch_metrics" directory with the name "epoch_metrics.parquet".
    """
    def __init__(self, filename='epoch_metrics'):
        self.filename = filename
        dm = DirectoryManager()
        self.fullpath = f'{dm.log_dir}/{self.filename}.parquet'

    def on_epoch_end(self, module, epoch):
        """
        Collects the epoch metrics and appends them as a new row to the metrics file.
        """
        metrics = {'epoch': epoch+1}
        
        if module.epoch_outputs is not None:
            metrics['trn_loss'] = module.epoch_outputs.get('train_loss', 0)
            metrics['trn_acc']  = module.epoch_outputs.get('train_accuracy', 0)
            metrics['trn_f1']   = module.epoch_outputs.get('train_f1_score_macro', 0)
            metrics['val_acc']  = module.epoch_outputs.get('accuracy', 0)
            metrics['val_f1']   = module.epoch_outputs.get('f1_score_macro', 0)
            metrics['val_loss'] = module.epoch_outputs.get('loss', 0)
        else:
            raise ValueError('module.epoch_outputs is None, cannot collect epoch metrics.')
        
        df_epoch = pd.DataFrame([metrics])
        if os.path.exists(self.fullpath):
            existing_df = pd.read_parquet(self.fullpath)
            df_new = pd.concat([existing_df, df_epoch], ignore_index=True)
            df_new.to_parquet(self.fullpath, index=False, compression='snappy')
        else:
            df_epoch.to_parquet(self.fullpath, index=False, compression='snappy')