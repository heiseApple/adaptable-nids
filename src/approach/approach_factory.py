from approach.ml_module import MLModule, ml_approaches
from approach.dl_module import DLModule, dl_approaches
from util.config import load_config
from callback.early_stopping_callback import EarlyStoppingCallback
from callback.save_outputs_callback import SaveOutputsCallback


def get_approach_type(approach_name):
    if approach_name in ml_approaches:
        return 'ml'
    elif approach_name in dl_approaches:
        return 'dl'
    else:
        raise ValueError(f"Approach '{approach_name}' not found in ML or DL approaches.")
    

def get_approach(approach_name, datamodule, **kwargs):
    callbacks = [SaveOutputsCallback()]
    cf = load_config()
    
    if approach_name in ml_approaches:
        return MLModule.get_approach(
            appr_name=approach_name,
            datamodule=datamodule,
            callbacks=callbacks,
            **kwargs
        )
    elif approach_name in dl_approaches:
        callbacks.extend([
            EarlyStoppingCallback(
                monitor=cf['es_monitor'],
                mode=cf['es_mode'],
                patience=cf['es_patience'],
                min_delta=cf['es_min_delta']
            )
        ])
        return DLModule.get_approach(
            appr_name=approach_name,
            datamodule=datamodule,
            callbacks=callbacks,
            **kwargs
        )   
    else:
        raise ValueError(f"Approach '{approach_name}' not found in ML or DL approaches.")