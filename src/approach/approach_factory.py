from approach.ml_module import MLModule, ml_approaches
from approach.dl_module import DLModule, dl_approaches
from util.config import load_config
from callback import (
    EarlyStopping,
    SaveOutputs,
    ModelCheckpoint
)


def get_approach_type(approach_name):
    if approach_name in ml_approaches:
        return 'ml'
    elif approach_name in dl_approaches:
        return 'dl'
    else:
        raise ValueError(f"Approach '{approach_name}' not found in ML or DL approaches.")
    

def get_approach(approach_name, datamodule, **kwargs):
    callbacks = [SaveOutputs()]
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
            EarlyStopping(
                monitor=cf['es_monitor'],
                mode=cf['es_mode'],
                patience=cf['es_patience'],
                min_delta=cf['es_min_delta']
            ),
            ModelCheckpoint(
                monitor=cf['mc_monitor'],
                mode=cf['mc_mode']
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