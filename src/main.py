import json
from argparse import ArgumentParser
 
from data.data import get_data_labels
from data.splitter import DatasetSplitter
from util.config import load_config
from util.results_evaluator import ResultsEvaluator
from util.directory_manager import DirectoryManager
from approach import (
    MLModule,
    RandomForest,
    XGB,
    KNN,
)


def main():
    ### 0 - PARSING INPUT
    cf = load_config()
    
    # Experiment args
    parser = ArgumentParser(conflict_handler='resolve', add_help=True) 
    parser = RandomForest.add_model_specific_args(parser)
    parser = XGB.add_model_specific_args(parser)
    parser = KNN.add_model_specific_args(parser)
    parser.add_argument('--seed', type=int, default=cf['seed'], help='Seed for reproducibility')
    parser.add_argument('--log-dir', type=str, default=cf['log_dir'], help='Log directory')
    parser.add_argument('--ml-appr', type=str, default=cf['ml_appr'], help='ML approach to use')
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
    dm = DirectoryManager(args.log_dir)
    with open(f'{dm.log_dir}/dict_args.json', 'w') as f:
        json.dump(dict_args, f)
    
    ### 1 - GET DATASET
    dataset = get_data_labels(
        dataset=args.dataset,
        num_pkts=args.num_pkts,
        fields=args.fields,
        is_flat=args.is_flat,
        seed=args.seed,
    )
    ds = DatasetSplitter(seed=args.seed, dataset=dataset)
    dataset_splits = ds.train_val_test_split()
    
    ### 2 - TRAIN AND TEST
    approach = MLModule.get_approach(
        ml_name=args.ml_appr, 
        **dict_args
    )
    approach.fit(dataset_splits['train'])
    approach.validate(dataset_splits['val'])
    approach.predict(dataset_splits['test'])
    
    re = ResultsEvaluator(dataset_name=args.dataset)
    for folder in ['test', 'val']:
        re.process_folder(folder)
    
if __name__ == '__main__':
    main()