from util.config import load_config

cf = load_config()
_BASE_DATA_PATH = cf['base_data_path']

dataset_config = {
    'iot23': {
        'path': f'{_BASE_DATA_PATH}/iot23_clean/dataset_20p_6f_576b_obf.parquet',
        'class_order': [12, 11, 1, 9, 4, 2, 0, 7, 5, 3, 8, 10, 6], 
        'fs_split': {  
            'train_classes': [12, 11, 1, 9, 4],  
            'val_classes': [2, 0, 7, 5],
            'test_classes': [3, 8, 10, 6]
        }
    },
    'cic2018': {  
        'path': f'{_BASE_DATA_PATH}/cic2018/cic2018_dataset_df_no_obf_20pkts_6feats_median_sampled_no_infiltration_clean_330ts.parquet',
        'class_order': [8, 9, 5, 1, 2, 0, 6, 10, 3, 7, 4, 11],  
        'fs_split': {  
            'train_classes': [8, 9, 5, 1, 2, 0], 
            'val_classes': [6, 10, 3],
            'test_classes': [7, 4, 11]
        },
        'label_column': 'LABEL_FULL', 
    }
}