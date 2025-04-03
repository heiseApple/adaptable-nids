import sys
import os


src_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

import json
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from types import SimpleNamespace

from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

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

    return (
        torch.cat(embeddings, dim=0).detach().cpu().numpy(), 
        torch.cat(labels, dim=0).detach().cpu().numpy()
    )


def main():
    # Read command-line arguments
    parser = ArgumentParser(conflict_handler='resolve', add_help=True)
    parser.add_argument('-a', '--args-path', type=str, default=None,
                        help='Path to dict_args.json')
    parser.add_argument('-c', '--ckpt-path', type=str, default=None,
                        help='Path to the .pt file containing the state of an approach')
    parser.add_argument('-o', '--output-fn', type=str, default='coords_2d.npz',
                        help='Output filename')
    parser.add_argument('-t', '--tsne', action='store_true', help='Compute TSNE embeddings')
    parser.add_argument('-s', '--silscore', action='store_true', help='Compute silhouette score')
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
    src_splits, trg_splits, _ = dm.get_dataset_splits()
    
    src_datamodule = src_splits.get_datamodule(**vars(exp_args))
    trg_datamodule = trg_splits.get_datamodule(**vars(exp_args))
    
    src_loader = src_datamodule.get_test_data()
    trg_loader = trg_datamodule.get_test_data()
    
    # Load the approach and network weights
    approach = get_approach(
        approach_name=exp_args.approach,
        fs_task=exp_args.k is not None,
        **vars(exp_args)
    )
    approach.load_checkpoint(script_args.ckpt_path) 
    print(f'Loaded approach state from {script_args.ckpt_path}')

    # Embed datasets
    src_embeddings, src_labels = embed_dataset(approach, src_loader, 'Embedding SRC dataset')
    trg_embeddings, trg_labels = embed_dataset(approach, trg_loader, 'Embedding TRG dataset')
    
    trg_labels = trg_labels + (src_labels.max().item() + 1)

    all_embeddings = np.concatenate([src_embeddings, trg_embeddings], axis=0)
    all_labels = np.concatenate([src_labels, trg_labels], axis=0)
    src_mask = np.zeros(len(all_labels), dtype=bool)
    src_mask[:len(src_labels)] = True

    # Apply TSNE
    if script_args.tsne:
        reducer = TSNE(n_components=2, verbose=1)
        coords_2d = reducer.fit_transform(all_embeddings)
    
    # Compute silhouette score
    if script_args.silscore:
        binary_labels = np.where(all_labels % 2 == 0, 0, 1)
        sil_score = silhouette_score(all_embeddings, binary_labels)
    
    # Save embeddings and labels to npz file
    np.savez(
        script_args.output_fn, 
        coords_2d=coords_2d if script_args.tsne else None, 
        src_labels=src_labels, 
        trg_labels=trg_labels,
        src_embeddings=src_embeddings,
        trg_embeddings=trg_embeddings,
        sil = sil_score if script_args.silscore else None,
    )

if __name__ == '__main__':
    main()
