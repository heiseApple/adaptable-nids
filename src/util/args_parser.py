from argparse import ArgumentParser

from data.data_module import DataModule
from util.config import load_config
from util.directory_manager import DirectoryManager
from approach import (
    RandomForest,
    XGB,
    KNN,
    Baseline,
    RFS,
    ADDA,
    get_approach_type,
    is_approach_usup,
)


def parse_arguments():
    cf = load_config()
    
    # Experiment args
    parser = ArgumentParser(conflict_handler='resolve', add_help=True) 
    parser = RandomForest.add_appr_specific_args(parser)
    parser = XGB.add_appr_specific_args(parser)
    parser = KNN.add_appr_specific_args(parser)
    parser = Baseline.add_appr_specific_args(parser)
    parser = RFS.add_appr_specific_args(parser)
    parser = ADDA.add_appr_specific_args(parser)
    parser = DataModule.add_argparse_args(parser)
    parser.add_argument('--seed', type=int, default=cf['seed'], help='Seed for reproducibility')
    parser.add_argument('--k-seed', type=int, default=cf['seed'], 
                        help='Seed used to sample the k samples in the few-shot case.')
    parser.add_argument('--gpu', action='store_true', default=cf['gpu'], help='Use GPU if available')
    parser.add_argument('--n-thr', type=int, default=cf['n_thr'], help='Number of threads')
    parser.add_argument('--log-dir', type=str, default=cf['log_dir'], help='Log directory')
    parser.add_argument('--n-tasks', type=int, default=cf['n_task'], choices=[1, 2], 
                        help='with 1 the model is trained on both src and trg dataset at the same time,\
                              with 2 the model is first trained on src then on trg')
    parser.add_argument('--approach', type=str, default=cf['approach'], help='ML or DL approach to use')
    parser.add_argument('--network', type=str, default=cf['network'], help='Network to use')
    parser.add_argument('--weights-path', type=str, default=cf['weights_path'], 
                        help='Path to the .pt file containing the weights for the network')
    parser.add_argument('--skip-t1', action='store_true', default=cf['skip_t1'], 
                        help='Skip the first task on src dataset, used only when n_task 2')
    parser.add_argument('--k', type=int, default=cf['k'], help='Number of shots for the target dataset')
    # Data args
    parser.add_argument('--src-dataset', type=str, default=cf['src_dataset'], 
                        help='Source dataset to use')
    parser.add_argument('--trg-dataset', type=str, default=cf['trg_dataset'], 
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
    
    args.appr_type = get_approach_type(args.approach) 
    args.is_appr_unsup = is_approach_usup(args.approach)
    
    if args.src_dataset is None:
        raise ValueError(f'Source Dataset is None')
    
    if args.trg_dataset==args.src_dataset:
        raise ValueError(f"Target dataset cannot be the same as source dataset: '{args.src_dataset}")
    
    if args.skip_t1 and args.n_tasks==2:
        print('WARNING: skipping task on src dataset')
    
    if args.appr_type == 'ml' and args.n_tasks > 1:
        raise ValueError('ML approaches do not support multiple tasks')
    
    if args.is_appr_unsup and args.n_tasks != 2:
        raise ValueError('Unsupervised approaches only support 2 tasks')
    
    # Create log dir
    DirectoryManager(args.log_dir)
    
    return args