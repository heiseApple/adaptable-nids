import argparse
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# List of keys to read from dict_args.json
DICT_ARGS_KEYS = [
    'seed',
    'approach',
    'src_dataset',
    'trg_dataset',
    'n_tasks',
    'num_pkts',
    'fields',
    'appr_type',
    'k',
    'adaptation_strat',
    'network',
]

# Metrics of interest for each report
REPORT_METRICS = [
    'accuracy',
    'precision_macro',
    'recall_macro',
    'f1_macro',
    'precision_micro',
    'recall_micro',
    'f1_micro'
]

def parse_report(report_path: Path) -> dict:
    """
    Reads the specified 'report_test.txt' file and returns a dictionary
    of metric_name -> metric_value for each line matching the pattern
    '<metric>: <value>' for metrics in REPORT_METRICS.
    """
    metrics = {}
    if not report_path.is_file():
        return metrics

    with report_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Look for lines of the form '<metric>: <value>'
            if ':' in line:
                parts = line.split(':', maxsplit=1)
                key = parts[0].strip()
                if key in REPORT_METRICS:
                    try:
                        value = float(parts[1].strip())
                        metrics[key] = value
                    except ValueError:
                        pass
    return metrics

def parse_experiments(root_dir: Path) -> pd.DataFrame:
    """
    Searches recursively under 'root_dir' for any 'dict_args.json' files.
    Assumes that each 'dict_args.json' is located inside an 'src' directory,
    and the experiment folder is the parent of 'src'.
    For each discovered file:
      - Reads 'dict_args.json' to fetch required fields.
      - Reads 'report_test.txt' from both 'src/detailed' and 'trg/detailed'.
      - Accumulates results into a DataFrame.
    """
    all_rows = []

    # Recursively find all 'dict_args.json' files.
    dict_args_files = list(root_dir.rglob('dict_args.json'))
    for dict_args_path in tqdm(dict_args_files, desc='Parsing experiments', total=len(dict_args_files)):
        # The experiment directory is assumed to be two levels above (parent of 'src').
        experiment_dir = dict_args_path.parent.parent  # e.g., /some/path/experiment_X
        if not experiment_dir.is_dir():
            continue

        # Read dict_args.json
        with dict_args_path.open('r', encoding='utf-8') as f:
            args_dict = json.load(f)

        # Build new_row from DICT_ARGS_KEYS
        new_row = {}
        for key in DICT_ARGS_KEYS:
            new_row[key] = args_dict.get(key)

        # Retrieve src_dataset and trg_dataset for use in parsing metrics
        src_dataset = new_row.get('src_dataset')
        trg_dataset = new_row.get('trg_dataset')

        # Parse report from src
        src_report = experiment_dir / 'src' / 'detailed' / 'report_test.txt'
        src_metrics = parse_report(src_report)
        if src_dataset:
            for metric_name, metric_value in src_metrics.items():
                new_row[f'src_{metric_name}'] = metric_value

        # Parse report from trg
        trg_report = experiment_dir / 'trg' / 'detailed' / 'report_test.txt'
        trg_metrics = parse_report(trg_report)
        if trg_dataset:
            for metric_name, metric_value in trg_metrics.items():
                new_row[f'trg_{metric_name}'] = metric_value

        all_rows.append(new_row)

    return pd.DataFrame(all_rows)

def main():
    parser = argparse.ArgumentParser(description='Parse experiment results by searching recursively for dict_args.json.')
    parser.add_argument('root_dir', type=str, 
                        help='Path to the directory containing all experiment folders.')
    parser.add_argument('--csv_output', type=str, default='results.csv', 
                        help='Optional path to save the consolidated DataFrame as a CSV file.')

    args = parser.parse_args()
    root_dir = Path(args.root_dir)

    df_results = parse_experiments(root_dir)

    print(f'\nOutput dataframe shape => {df_results.shape}\n', df_results.head(3))

    if args.csv_output:
        df_results.to_csv(args.csv_output, index=False)
        print(f'Results have been saved to {args.csv_output}')

if __name__ == '__main__':
    main()