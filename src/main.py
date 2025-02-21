from util.args_parser import parse_arguments
from util.seed import seed_everything

from util.logger import Logger
from data.data_manager import DataManager
from trainer.trainer import Trainer


def main():
    # 1. Parse arguments
    args = parse_arguments()
    
    # 2. Set the seed
    seed_everything(args.seed)
    
    # 3. Init the DataManager and load the data
    data_manager = DataManager(args)
    data_manager.load_datasets()
    
    # 4. Initialize the Trainer and start the run
    trainer = Trainer(args, data_manager)
    trainer.run()
    
    # 5. Final logging (metrics, graphs, etc.)
    logger = Logger(src_dataset_name=args.src_dataset, trg_dataset_name=args.trg_dataset)
    logger.process_results()
    logger.plot_per_epoch_metrics()


if __name__ == '__main__':
    main()
