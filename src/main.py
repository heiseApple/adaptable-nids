import json
from argparse import ArgumentParser
 
from data.data import get_data_labels
from data.datamodule import DataModule
from data.splitter import DatasetSplitter
from util.config import load_config
from util.logger import Logger
from util.directory_manager import DirectoryManager
from util.seed import seed_everything
from approach import (
    RandomForest,
    XGB,
    KNN,
    Scratch,
    get_approach,
    get_approach_type
)


def main():
    ### 0 - PARSING INPUT
    cf = load_config()
    
    # Experiment args
    parser = ArgumentParser(conflict_handler='resolve', add_help=True) 
    parser = RandomForest.add_appr_specific_args(parser)
    parser = XGB.add_appr_specific_args(parser)
    parser = KNN.add_appr_specific_args(parser)
    parser = Scratch.add_appr_specific_args(parser)
    parser.add_argument('--seed', type=int, default=cf['seed'], help='Seed for reproducibility')
    parser.add_argument('--gpu', action='store_true', default=cf['gpu'], help='Use GPU if available')
    parser.add_argument('--log-dir', type=str, default=cf['log_dir'], help='Log directory')
    parser.add_argument('--approach', type=str, default=cf['approach'], help='ML or DL approach to use')
    parser.add_argument('--network', type=str, default=cf['network'], help='Network to use')
    # Data args
    parser.add_argument('--dataset', type=str, default=cf['dataset'], help='Dataset to use')
    parser.add_argument('--is-flat', action='store_true', default=cf['is_flat'],
                        help='Flat the PSQ input')
    parser.add_argument('--num-pkts', type=int, default=cf['num_pkts'], 
                        help='Number of packets to consider in each biflow')
    parser.add_argument('--fields', type=str, default=cf['fields'],  
                        choices=['PL', 'IAT', 'DIR', 'WIN', 'FLG', 'TTL'],
                        help='Field or fields used (default=%(default)s)', 
                        nargs='+', metavar='FIELD')
    
    args = parser.parse_args()
    dict_args = vars(args)
    
    seed_everything(args.seed)
    
    ### 1 - GET DATASET
    dataset = get_data_labels(
        dataset=args.dataset,
        num_pkts=args.num_pkts,
        fields=args.fields,
        is_flat=args.is_flat,
        seed=args.seed,
    )
    ds = DatasetSplitter(seed=args.seed, dataset=dataset)
    dataset_splits, dict_args['num_classes'] = ds.train_val_test_split()
    datamodule = DataModule(
        train_dataset=dataset_splits['train'],
        val_dataset=dataset_splits['val'],
        test_dataset=dataset_splits['test'],
        appr_type=get_approach_type(args.approach),
    )
    
    dm = DirectoryManager(args.log_dir)
    with open(f'{dm.log_dir}/dict_args.json', 'w') as f:
        json.dump(dict_args, f)
    
    ### 2 - TRAIN AND TEST
    approach = get_approach(approach_name=args.approach, datamodule=datamodule, **dict_args)
        
    approach.fit()
    approach.validation()
    approach.test()
    
    logger = Logger(dataset_name=args.dataset)
    for folder in ['test', 'val']:
        logger.process_folder(folder)
    logger.plot_metrics()
    
if __name__ == '__main__':
    main()