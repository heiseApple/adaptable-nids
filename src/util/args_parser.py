from argparse import ArgumentParser

from util.config import load_config
from approach import (
    RandomForest,
    XGB,
    KNN,
    Baseline,
    get_approach_type
)


def parse_arguments():
    cf = load_config()
    
    # Experiment args
    parser = ArgumentParser(conflict_handler='resolve', add_help=True) 
    parser = RandomForest.add_appr_specific_args(parser)
    parser = XGB.add_appr_specific_args(parser)
    parser = KNN.add_appr_specific_args(parser)
    parser = Baseline.add_appr_specific_args(parser)
    parser.add_argument('--seed', type=int, default=cf['seed'], help='Seed for reproducibility')
    parser.add_argument('--gpu', action='store_true', default=cf['gpu'], help='Use GPU if available')
    parser.add_argument('--log-dir', type=str, default=cf['log_dir'], help='Log directory')
    parser.add_argument('--n-tasks', type=int, default=cf['n_task'], choices=[1, 2], 
                        help='with 1 the model is trained on both src and trg dataset at the same time,\
                              with 2 the model is first trained on src then on dst')
    parser.add_argument('--approach', type=str, default=cf['approach'], help='ML or DL approach to use')
    parser.add_argument('--network', type=str, default=cf['network'], help='Network to use')
    # Data args
    parser.add_argument('--src-dataset', type=str, default=cf['src-dataset'], 
                        help='Source dataset to use')
    parser.add_argument('--trg-dataset', type=str, default=cf['trg-dataset'], 
                        help='Target dataset to use')
    parser.add_argument('--is-flat', action='store_true', default=cf['is_flat'],
                        help='Flat the PSQ input')
    parser.add_argument('--num-pkts', type=int, default=cf['num_pkts'], 
                        help='Number of packets to consider in each biflow')
    parser.add_argument('--fields', type=str, default=cf['fields'],  
                        choices=['PL', 'IAT', 'DIR', 'WIN', 'FLG', 'TTL'],
                        help='Field or fields used (default=%(default)s)', 
                        nargs='+', metavar='FIELD')
    
    args = parser.parse_args()
    
    if args.src_dataset is None:
        raise ValueError(f'Source Dataset is None')
    
    if args.trg_dataset==args.src_dataset:
        raise ValueError(f"Target dataset cannot be the same as source dataset: '{args.src_dataset}")
    
    args.appr_type = get_approach_type(args.approach) 

    return args