import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from data.dataset_config import dataset_config
from util.directory_manager import DirectoryManager


class Logger:
    """
    A class for evaluating prediction results by:
    - Loading label and prediction data from .npz files.
    - Computing standard classification metrics including accuracy, precision,
        recall, and F1-score (both macro and micro averages).
    - Saving detailed evaluation reports to text files.
    - Generating and saving confusion matrices (as CSV data and a normalized heatmap in PDF format).
    - Plotting epoch metrics stored in a Parquet file as a line plot.
    Attributes:
        subfolders (tuple): A tuple of subfolder names (e.g., ('test', 'val')) indicating which
                            folders to process.
        detailed_dir (str): The directory where detailed reports and plots are saved.
        log_dir (str): The directory where log data is stored.
    """
    def __init__(self, subfolders=('test', 'val'), dataset_name=None):
        self.subfolders = subfolders
        
        dm = DirectoryManager()
        self.detailed_dir = dm.mkdir('detailed')
        self.log_dir = dm.log_dir
        self.dataset_name = dataset_name
        
        print('='*100)


    def _load_data(self, folder_path):
        labels_data = np.load(f'{folder_path}/labels.npz')
        preds_data = np.load(f'{folder_path}/preds.npz')
        labels = labels_data[labels_data.files[0]]
        preds = preds_data[preds_data.files[0]]
        return labels, preds


    def _compute_metrics(self, labels, preds):
        metrics_dict = {}
        metrics_dict['accuracy'] = accuracy_score(labels, preds)
        metrics_dict['precision_macro'] = precision_score(labels, preds, average='macro', zero_division=0)
        metrics_dict['recall_macro'] = recall_score(labels, preds, average='macro', zero_division=0)
        metrics_dict['f1_macro'] = f1_score(labels, preds, average='macro', zero_division=0)
        metrics_dict['precision_micro'] = precision_score(labels, preds, average='micro', zero_division=0)
        metrics_dict['recall_micro'] = recall_score(labels, preds, average='micro', zero_division=0)
        metrics_dict['f1_micro'] = f1_score(labels, preds, average='micro', zero_division=0)
        metrics_dict['classification_report'] = classification_report(labels, preds, zero_division=0)
        return metrics_dict
    
    
    def _save_report(self, metrics, folder_name):
        with open(f'{self.detailed_dir}/report_{folder_name}.txt', 'w') as f:
            f.write('Metrics Report\n')
            f.write('=========================\n')
            for key, value in metrics.items():
                if key == 'classification_report':
                    f.write(f'\n{key}:\n{value}\n')
                else:
                    f.write(f'{key}: {value}\n')
        print(f'Report saved => {self.detailed_dir}/report_{folder_name}.txt')
        
        
    def _generate_confusion_matrices(self, labels, preds, folder_name):
        cm = confusion_matrix(labels, preds)
        np.savetxt(
            f'{self.detailed_dir}/confusion_matrix_{folder_name}.csv', cm, delimiter=',', fmt='%d'
        )
        cm_norm = confusion_matrix(labels, preds, normalize='true')
        
        # Read the classes ecoding json file
        dc = dataset_config[self.dataset_name]
        with open(f"{os.path.dirname(dc['path'])}/label_conv.json", 'r') as f:
            label_conv = json.load(f)
        classes = label_conv.keys()
            
        # Plot CM with seaborn
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=classes, yticklabels=classes, cmap='viridis')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Normalized Confusion Matrix')
        plt.savefig(f'{self.detailed_dir}/confusion_matrix_{folder_name}.pdf', bbox_inches='tight')
        plt.close()
        
        
    def process_folder(self, folder_name):
        """
        Processes a specific folder by:
        1. Loading the label and prediction data from the log directory for the given folder.
        2. Computing various classification metrics.
        3. Saving a detailed report of the metrics to a text file.
        4. Generating and saving both raw and normalized confusion matrices, with the normalized version plotted as a heatmap.
        Parameters:
            folder_name (str): The name of the folder (located within the log directory) containing the evaluation data.
        """
        if folder_name not in self.subfolders:
            return
        labels, preds = self._load_data(f'{self.log_dir}/{folder_name}')
        metrics = self._compute_metrics(labels, preds)
        self._save_report(metrics, folder_name)
        self._generate_confusion_matrices(labels, preds, folder_name)
        
        
    def _plot_metric_group(self, df, group_cols, ylabel, output_filename):

        max_epoch = df.shape[0]
        plt.figure(figsize=(max_epoch*0.20, 5))
        for col in group_cols:
            plt.plot(df['epoch'], df[col], marker='o', label=col)
            
        plt.xlabel('Epochs')
        plt.ylabel(ylabel)
        plt.grid(linestyle='--', color='gray')
        plt.xticks(np.arange(1, max_epoch + 1, 1), fontsize=8, rotation=90)
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        plt.savefig(f'{self.detailed_dir}/{output_filename}', bbox_inches='tight')
        plt.close()        
        
        
    def plot_metrics(self, filename='epoch_metrics'):
        """
        Reads a Parquet file containing epoch metrics and plots a line plot
        of the metrics over epochs. The x-axis represents epochs.
        """
        dm = DirectoryManager()
        file_path = f'{dm.log_dir}/{filename}.parquet'
        if not os.path.exists(file_path):
            return
        df = pd.read_parquet(file_path)
        
        metrics_columns = [col for col in df.columns if col != 'epoch']
        loss_cols = [col for col in metrics_columns if 'loss' in col.lower()]
        other_cols = [col for col in metrics_columns if col not in loss_cols]
        
        self._plot_metric_group(
            df, loss_cols, ylabel='Loss Value', output_filename='epoch_loss.pdf')
        self._plot_metric_group(
            df, other_cols, ylabel='Metric Value', output_filename='epoch_metrics.pdf')
        