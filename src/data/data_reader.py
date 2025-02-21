import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from data.dataset_config import dataset_config
from util.config import load_config


def get_data_labels(dataset, num_pkts, fields, is_flat, seed):
    """
    Preprocesses a dataset and returns the input features and labels.

    Parameters:
        dataset (str): The name of the dataset to be used.
        num_pkts (int): The number of packets to consider.
        fields (list): List of fields to include in the input features.
        is_flat (bool): If True, returns flattened input features (for ML models).
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple (x, y) where x is the input features and y are the labels.
    """
    dataset_dict = dict()
    dc = dataset_config[dataset]
    full_path = dc['path']
    label_column = dc.get('label_column', 'LABEL')
    
    p = Path(full_path)
    prep_df_path = p.parent / f'{p.stem}_{label_column}_prep{seed}{p.suffix}'
    
    if not prep_df_path.exists():
        # First time reading the dataset 
        print(f'Processing {dataset} dataframe...')
        
        cf = load_config()
        
        df = pd.read_parquet(full_path)
        df = _preprocess_dataframe(df, label_column, p.parent, cf)
        df.to_parquet(prep_df_path)
    else:
        # Already pre-processed
        print(f'WARNING: using pre-processed dataframe for {dataset}.')
        df = pd.read_parquet(prep_df_path)
            
    # Compute PSQ input N_p x F
    data_series = df[[f'SCALED_{f}' for f in fields]].apply(
        lambda row: _process_row(row, num_pkts), axis=1
    )
    data = np.concatenate(data_series.tolist(), axis=0).astype(np.float32) 
    data = np.expand_dims(data, axis=1)
    
    dataset_dict['labels'] = np.array([label for label in df['ENC_LABEL']], dtype=np.int64)
    if is_flat:
        dataset_dict['data'] = np.array([np.ravel(a.T) for a in data])
        return dataset_dict
    dataset_dict['data'] = data
    return dataset_dict


def _preprocess_dataframe(df, label_column, parent_dir, config):
    """
    Preprocess the dataframe by performing label encoding, field padding, and scaling.
    """
    all_fields = config['all_fields']
    pad_value = config['pad_value']
    pad_value_dir = config['pad_value_dir']

    # Label encoding
    le = LabelEncoder()
    le.fit(df[label_column])
    df['ENC_LABEL'] = le.transform(df[label_column])
    
    # Save encoding informations
    label_conv = dict(zip(le.classes_, le.transform(le.classes_).tolist()))
    with open(parent_dir / 'label_conv.json', 'w') as f:
        json.dump(label_conv, f)
        
    processed_fields = []
    for f in all_fields:
        
        if f not in df.columns:
            continue
        
        # Field padding
        pv = pad_value_dir if f == 'DIR' else pad_value
        df[f] = df[[f, 'FEAT_PAD']].apply(
            lambda x: np.concatenate((x[f], [pv] * x['FEAT_PAD'])), axis=1)
        
        # Field scaling
        mms = MinMaxScaler((0, 1))
        mms.fit(np.concatenate(df[f].values, axis=0).reshape(-1, 1))
        df[f'SCALED_{f}'] = df[f].apply(
            lambda x: mms.transform(x.reshape(-1, 1)).reshape(-1))
        processed_fields.append(f)
        
    # Pick only preprocessed fields and encoded labels
    return df[[f'SCALED_{f}' for f in processed_fields] + ['ENC_LABEL']]


def _process_row(row, num_pkts):
    """
    Process a single row by slicing each field to num_pkts, stacking them,
    transposing the result, and adding an extra dimension.
    """
    # Convert each field's data to a numpy array and slice to num_pkts
    field_arrays = [np.array(f)[:num_pkts] for f in row]
    # Stack along a new axis to get shape (F, num_pkts)
    stacked = np.stack(field_arrays, axis=0)
    # Transpose to (num_pkts, F) and add a new axis at the beginning => (1, num_pkts, F)
    return np.expand_dims(stacked.T, axis=0)