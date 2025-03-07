# adaptable-nids

## Installation

1. Place the dataset according to the path defined in [`config.yaml`](config.yaml) under `base_data_path`.  
   Each dataset must then be placed in its own folder as defined in [`dataset_config.py`](src/data/dataset_config.py).

2. It is recommended to use `virtualenv` to create an isolated Python environment:
    ```bash
    virtualenv venv
    source venv/bin/activate
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
    It is also recommended to install [GNU parallel](https://www.gnu.org/software/parallel/).

## How To Use It

Navigate to the `src` directory and execute the main script:
```bash
cd src
python main.py
```
You can then append any of the following options to your command.  
Unless otherwise specified below, the default values are taken from [`config.yaml`](config.yaml).
The parsing logic is defined in [`args_parser.py`](src/util/args_parser.py).

### General Arguments

- `--seed [int]`: Seed for reproducibility.  
- `--k-seed [int]`: Seed used to sample the k shots in the few-shot case.  
- `--gpu`: Use GPU if available.  
- `--n-thr [int]`: Number of threads.  
- `--log-dir [str]`: Log directory path.  
- `--n-tasks [1 or 2]`:  
   `1`: The model is trained on both source and target datasets at the same time.  
   `2`: The model is first trained on the source dataset, then on the target dataset.  
- `--network [str]`: Network to use. The value must match the `.py` filename in [`src/network/`](src/network/) that implements the network (e.g., `lopez17cnn`). 
- `--weights-path [str]`: Path to the `.pt` file containing the weights for the network.  
- `--skip-t1`: Skip the first task on the source dataset (used only when `--n-tasks 2`).  
- `--k [int]`: Number of shots for the target dataset. If not specified, the entire training partition of the target dataset is used.

The [`config.yaml`](config.yaml) file contains additional parameters (e.g., for early stopping, checkpointing, etc.).

### Data-Related Arguments

In addition to the standard dataset selection arguments, the following data-related parameters can be used for data loading and processing.

- `--src-dataset [str]`: Source dataset to use.  
- `--trg-dataset [str]`: Target dataset to use.  
- `--is-flat`: Flatten the PSQ input (used for ML approaches).  
- `--num-pkts [int]`: Number of packets to consider in each biflow.  
- `--fields [FIELD] ...`: Fields used among `['PL', 'IAT', 'DIR', 'WIN', 'FLG', 'TTL']`.  
  You can specify multiple fields (e.g., `--fields PL IAT`).  

The following options are defined in [`data_module.py`](src/data/data_module.py):

- `--batch-size [int]`: Batch size for training.  
- `--adapt-batch-size [int]`: Batch size for adaptation.  
- `--num-workers [int]`: Number of worker threads for data loading.  
- `--pin-memory`: Enable pinned memory for faster data transfer to GPU.

### Approach Arguments

The two main modules responsible for training and validation are:  

- `MLModule` in [`ml_module.py`](src/approach/ml_module.py) for ML approaches.  
- `DLModule` in [`dl_module.py`](src/approach/dl_module.py) for DL approaches.  

To execute a specific approach located in [`src/approach/`](src/approach/), set the `--approach` argument to the corresponding `.py` file name.  

Additionally, each approach defines its own set of arguments, declared within the respective class. These approach-specific arguments are listed below:

#### K-Nearest Neighbors (KNN) – [`knn.py`](src/approach/knn.py)

- `--knn-n-neighbors [int]`: Number of neighbors to use.  
- `--knn-weights [str]`: Weight function used in prediction.  
- `--knn-p [int]`: Power parameter for the Minkowski metric.  
- `--knn-metric [str]`: Distance metric to use.

#### Random Forest (RF) – [`random_forest.py`](src/approach/random_forest.py)

- `--criterion [str]`: Function to measure the quality of a split.  
- `--rf-n-estimators [int]`: Number of trees in the forest.  
- `--rf-max-depth [int]`: Maximum depth of the trees. 

#### XGBoost (XGB) – [`xgb.py`](src/approach/xgb.py)

- `--xgb-n-estimators [int]`: Number of boosting rounds.  
- `--xgb-max-depth [int]`: Maximum tree depth for base learners.  
- `--eval-metric [str]`: Evaluation metric for validation data.  

#### Baseline (FineTuning and Freezing) – [`baseline.py`](src/approach/baseline.py)

- `--adaptation-strat [str]`: Strategy for adapting the model (`finetuning` or `freezing`).  
- `--adapt-lr [float]`: Learning rate for adaptation.  
- `--adapt-epochs [int]`: Number of epochs for adaptation.  

#### RFS (Rethinking Few-Shot) – [`rfs.py`](src/approach/rfs.py)

- `--alpha [float]`: Weighting factor for distillation loss.  
- `--gamma [float]`: Weighting factor for classification loss.  
- `--is-distill`: Enables knowledge distillation.  
- `--kd-t [float]`: Temperature for distillation loss.  
- `--teacher-path [str]`: Path to the pretrained teacher model in `.pt` extention. 

## Project Structure

```plaintext
adaptable-nids/
├── config.yaml
├── requirements.txt
├── parse_experiments.py
└── src/
    ├── main.py
    ├── run_experiments.sh
    ├── approach/
    ├── callback/
    ├── data/
    ├── module/
    ├── network/
    ├── trainer/
    ├── util/
```

This project is organized into multiple directories, each serving a specific purpose.

### Approach

Located in [`src/approach/`](src/approach/), this directory contains implementations of different ML and DL approaches. Each approach defines its own training, validation, and inference logic, and can be selected via the `--approach` argument.

### Callback

Located in [`src/callback/`](src/callback/), this directory includes callback functions that modify training behavior. These callbacks handle tasks such as early stopping, model checkpointing, logging outputs, and more.

### Data

Located in [`src/data/`](src/data/), this directory is responsible for dataset management, including loading, preprocessing, and configuration. It provides utilities to read datasets, set up batch sizes, and define dataset-related parameters.

### Module

Located in [`src/module/`](src/module/), this directory contains core components related to DL models. It includes implementations for custom loss functions, teacher-student learning strategies, and neural network heads.

### Network

Located in [`src/network/`](src/network/), this directory defines different neural network architectures used in the project. It provides a selection of predefined networks and a factory method for dynamically choosing a network based on configuration.

### Trainer

Located in [`src/trainer/`](src/trainer/), this directory contains the main training pipeline. It manages the optimization process, evaluation, and model adaptation strategies.

### Util

Located in [`src/util/`](src/util/), this directory includes utility functions that support the overall framework. It handles configuration management, argument parsing, logging, directory creation, and setting random seeds for reproducibility.

## Execution of Experiments

Experiments can be executed in two ways:

### 1. Direct Execution via `main.py`
You can manually run experiments by navigating to the `src` directory and executing:
```bash
python main.py --src-dataset <source> --trg-dataset <target> --approach <approach> --seed <seed> [other options]
```
This allows full control over individual experiment parameters.

### 2. Batch Execution via `run_experiments.sh`

For running multiple experiments in a combinatorial manner, use the [`run_experiments.sh`](src/run_experiments.sh) script:
You can manually run experiments by navigating to the `src` directory and executing:
```bash
./run_experiments.sh --src-dataset sd1,sd2 --trg-dataset td1,td2 \
    --seed 0-10 --approach knn,xgb --cpu 4 --log-keyword test
```
This script automatically generates experiment combinations based on provided parameters and runs them in parallel if [GNU parallel](https://www.gnu.org/software/parallel/) is available.
Otherwise, it falls back to `xargs`.

The project includes [`parse_experiments.py`](src/parse_experiments.py), a utility script that processes experiment logs and generates `.csv` files summarizing key metrics and characteristics of the experiments conducted within a specified target folder.

## Acknowledgement

We thank the following open-source implementations that were used in this work:

- [LibFewShot](https://github.com/RL-VIG/LibFewShot)
- [GNU parallel](https://www.gnu.org/software/parallel/)

## Citation
If you use this framework, please cite:
```
@article{xx,
  title = 
}
```