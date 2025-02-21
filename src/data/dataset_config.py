from util.config import load_config

cf = load_config()
_BASE_DATA_PATH = cf['base_data_path']

dataset_config = {
    'iot23': {
        'path': (
            f'{_BASE_DATA_PATH}/iot23_clean/'
            'iot23_dataset_df_obf_median_sampled_20pkts_6feats_rect-dir_botnet_clean_mirage_class_over_10.parquet'
        ),
        'label_column': 'LABEL-bin',
    },
    'cic2018': {
        'path': (
            f'{_BASE_DATA_PATH}/cic2018/'
            'cic2018_dataset_df_no_obf_20pkts_6feats_median_sampled_no_infiltration_clean_330ts.parquet'
        ),
        'label_column': 'LABEL-bin',
    },
    'insdn': {
        'path': (
            f'{_BASE_DATA_PATH}/in_sdn/'
            'in_sdn_20pkts_6f_net_1024bytes_infsTO_network_class_over_50.parquet'
        ),
        'label_column': 'LABEL-bin',
    },
    'edgeiiot': {
        'path': (
            f'{_BASE_DATA_PATH}/edge_iiot/'
            'edge-iiot_100pkts_6f_1p-mt100k_benign_class_clean.parquet'
        )
    },
}
