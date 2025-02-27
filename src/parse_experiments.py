import argparse
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# List of keys to read from dict_args.json
DICT_ARGS_KEYS = [
    'seed',
    'approach',
    'src_dataset',
    'trg_dataset',
    'n_tasks',
    'num_pkts',
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

def aggregate_metrics(df):
    grouping_keys = [k for k in DICT_ARGS_KEYS if k != 'seed']
    
    metric_cols = []
    for m in REPORT_METRICS:
        for t in ['src', 'trg']:
            metric_cols.append(f'{t}_{m}')

    grouped = df.groupby(grouping_keys)[metric_cols].agg(['mean', 'std']).reset_index()
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    
    count_df = df.groupby(grouping_keys).size().reset_index(name='n_rep')
    grouped = grouped.merge(count_df, on=grouping_keys, how='left')
    
    return grouped

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
    Recursively searches for 'dict_args.json' under root_dir.
    Each 'dict_args.json' is assumed to be in a 'src' folder,
    with the experiment folder as the parent of 'src'.
    
    For each found file:
      - Read 'dict_args.json' to get arguments.
      - Look for 'report_test.txt' in 'src/detailed' and 'trg/detailed'.
      - Build a row in a DataFrame with argument values and metrics.
    """
    all_rows = []

    dict_args_files = list(root_dir.rglob('dict_args.json'))
    for dict_args_path in tqdm(dict_args_files, desc='Parsing experiments', total=len(dict_args_files)):
        experiment_dir = dict_args_path.parent.parent
        if not experiment_dir.is_dir():
            continue

        # Read dict_args.json
        with dict_args_path.open('r', encoding='utf-8') as f:
            args_dict = json.load(f)

        new_row = {}
        for key in DICT_ARGS_KEYS:
            new_row[key] = args_dict.get(key)

        # Parse metrics from src/detailed/report_test.txt
        src_report = experiment_dir / 'src' / 'detailed' / 'report_test.txt'
        src_metrics = parse_report(src_report)
        if new_row.get('src_dataset'):
            for metric_name, metric_value in src_metrics.items():
                new_row[f'src_{metric_name}'] = metric_value

        # Parse metrics from trg/detailed/report_test.txt
        trg_report = experiment_dir / 'trg' / 'detailed' / 'report_test.txt'
        trg_metrics = parse_report(trg_report)
        if new_row.get('trg_dataset'):
            for metric_name, metric_value in trg_metrics.items():
                new_row[f'trg_{metric_name}'] = metric_value

        all_rows.append(new_row)

    return pd.DataFrame(all_rows)

def main():
    parser = argparse.ArgumentParser(description='Parse experiment results by searching recursively for dict_args.json.')
    parser.add_argument('root_dir', type=str, 
                        help='Path to the directory containing all experiment folders.')
    parser.add_argument('--csv-output', type=str, default='results.csv', 
                        help='Optional path to save the full results DataFrame as CSV.')
    parser.add_argument('--csv-output-agg', type=str, default='results_agg.csv',
                        help='Optional path to save the aggregated results DataFrame as CSV.')

    args = parser.parse_args()
    root_dir = Path(args.root_dir)

    df_results = parse_experiments(root_dir)

    print(f'\nOutput dataframe shape => {df_results.shape}\n', df_results.head(3))

    df_agg = aggregate_metrics(df_results)
    print(f'\nAggregated DF shape => {df_agg.shape}\n', df_agg.head(3))

    if args.csv_output:
        df_results.to_csv(args.csv_output, index=False)
        print(f'[INFO] Full results saved to: {args.csv_output}')

    if args.csv_output_agg:
        df_agg.to_csv(args.csv_output_agg, index=False)
        print(f'[INFO] Aggregated results saved to: {args.csv_output_agg}')

if __name__ == '__main__':
    main()
