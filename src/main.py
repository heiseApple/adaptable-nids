import json
from argparse import ArgumentParser
 
from data.data import get_data_labels
from util.config import load_config
from util.directory_manager import DirectoryManager

def main():
    ### 0 - PARSING INPUT
    cf = load_config()
    
    parser = ArgumentParser(conflict_handler='resolve', add_help=True) 
    parser.add_argument('--seed', type=int, default=cf['seed'], help='Seed for reproducibility')
    parser.add_argument('--log-dir', type=str, default=cf['log_dir'], help='Log directory') # in appr
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
    
    ### 1 - GET DATASET AND LOADERS
    data, labels = get_data_labels(
        dataset=args.dataset,
        num_pkts=args.num_pkts,
        fields=args.fields,
        is_flat=args.is_flat,
        seed=args.seed,
    )
    print(data.shape, labels.shape)
    print(data[0], labels[0])
    
    
if __name__ == '__main__':
    main()