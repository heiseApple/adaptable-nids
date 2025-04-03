import sys
import os


src_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

import json
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from types import SimpleNamespace

from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

import torch
from torch.utils.data import DataLoader, TensorDataset

from data.data_manager import DataManager
from approach.approach_factory import get_approach
from util.seed import seed_everything


def embed_dataset(approach, loader, desc):
    embeddings, labels = [], []
    approach.net.eval()
    loop = tqdm(loader, desc=desc)

    for batch_x, batch_y in loop:
        with torch.no_grad():
            batch_x, batch_y = batch_x.to(approach.device), batch_y.to(approach.device)
            _, batch_emb = approach.net(batch_x, return_feat=True)
            embeddings.append(batch_emb)
            labels.append(batch_y)

    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)


def main():
    # Read command-line arguments
    parser = ArgumentParser(conflict_handler='resolve', add_help=True)
    parser.add_argument('-a', '--args-path', type=str, default=None,
                        help='Path to dict_args.json')
    parser.add_argument('-w', '--weights-path', type=str, default=None,
                        help='Path to the .pt file containing the network weights')
    parser.add_argument('-d', '--data-percent', type=float, default=1.0,
                        help='Percentage of data to use (stratified sampling)')
    parser.add_argument('-o', '--output-fn', type=str, default='coords_2d.npz',
                        help='Output filename')
    script_args = parser.parse_args()

    # Load experiment arguments
    with open(script_args.args_path) as f:
        exp_args = SimpleNamespace(**json.load(f))

    if exp_args.appr_type == 'ml':
        raise ValueError('Compatible only with DL approaches.')

    # Set random seed
    seed_everything(exp_args.seed)

    # Prepare data
    dm = DataManager(exp_args)
    dm.load_datasets()
    
    # Apply stratified sampling if needed
    if script_args.data_percent < 1.0:
        src_data = dm.src_dataset['data']
        src_labels = dm.src_dataset['labels']
        src_data, _, src_labels, _ = train_test_split(
            src_data, src_labels, 
            train_size=script_args.data_percent,
            stratify=src_labels,
            random_state=0
        )
        dm.src_dataset['data'] = src_data
        dm.src_dataset['labels'] = src_labels

        trg_data = dm.trg_dataset['data']
        trg_labels = dm.trg_dataset['labels']
        trg_data, _, trg_labels, _ = train_test_split(
            trg_data, trg_labels,
            train_size=script_args.data_percent,
            stratify=trg_labels,
            random_state=0
        )
        dm.trg_dataset['data'] = trg_data
        dm.trg_dataset['labels'] = trg_labels

    src_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(dm.src_dataset['data']).float(),
            torch.from_numpy(dm.src_dataset['labels']).float()
        ),
        batch_size=exp_args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    trg_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(dm.trg_dataset['data']).float(),
            torch.from_numpy(dm.trg_dataset['labels']).float()
        ),
        batch_size=exp_args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # Load the approach and network weights
    approach = get_approach(
        approach_name=exp_args.approach,
        fs_task=exp_args.k is not None,
        **vars(exp_args)
    )
    approach.net.load_weights(script_args.weights_path)
    print(f'Loaded network weights from {script_args.weights_path}')

    # Embed datasets
    src_embeddings, src_labels = embed_dataset(approach, src_loader, 'Embedding SRC dataset')
    trg_embeddings, trg_labels = embed_dataset(approach, trg_loader, 'Embedding TRG dataset')
    
    trg_labels = trg_labels + (src_labels.max().item() + 1)

    all_embeddings = np.concatenate([src_embeddings, trg_embeddings], axis=0)
    all_labels = np.concatenate([src_labels, trg_labels], axis=0)
    src_mask = np.zeros(len(all_labels), dtype=bool)
    src_mask[:len(src_labels)] = True

    # Apply TSNE
    reducer = TSNE(n_components=2, verbose=1)
    coords_2d = reducer.fit_transform(all_embeddings)
    
    # Compute silhouette score
    binary_labels = np.where(all_labels % 2 == 0, 0, 1)
    sil_score = silhouette_score(all_embeddings, binary_labels)
    
    # Save embeddings and labels to npz file
    np.savez(
        script_args.output_fn, 
        coords_2d=coords_2d, 
        src_labels=src_labels.detach().cpu().numpy(), 
        trg_labels=trg_labels.detach().cpu().numpy(),
        src_embeddings=src_embeddings.detach().cpu().numpy(),
        trg_embeddings=trg_embeddings.detach().cpu().numpy(),
        sil = sil_score,
    )

if __name__ == '__main__':
    main()
