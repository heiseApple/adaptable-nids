#!/bin/bash
# run_experiments.sh
# This script runs experiments in a combinatorial fashion.
# It accepts:
#   --dataset (-d)   : a comma-separated list of datasets
#   --seed (-s)      : a comma-separated list of seeds
#   --approach (-a)  : a comma-separated list of approaches
#   --is-flat (-f)   : a flag indicating flat structure (if present, add to command)
#   --cpu (-c)       : an integer for the number of cores to use
#   --extra-args (-e): additional arguments to pass to the python script

usage() {
    echo "Usage: $0 --dataset <ds1,ds2,...> --seed <s1,s2,...> --approach <appr1,appr2,...> [--is-flat] --cpu <num_cores> [--extra-args <args>]"
    exit 1
}

# Default values
DATASETS=""
SEEDS=""
APPROACHES=""
IS_FLAT=0
CPU_CORES=""
EXTRA_ARGS=""

# Parse arguments
# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -d|--dataset)
            DATASETS="$2"
            shift ;;
        -s|--seed)
            SEEDS="$2"
            shift ;;
        -a|--approach)
            APPROACHES="$2"
            shift ;;
        -f|--is-flat)
            IS_FLAT=1 ;;
        -c|--cpu)
            CPU_CORES="$2"
            shift ;;
        -e|--extra-args)
            shift
            EXTRA_ARGS=()
            while [[ "$#" -gt 0 ]] && [[ "$1" != -* ]]; do
                EXTRA_ARGS+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown parameter passed: $1"
            usage ;;
    esac
    shift
done

if [[ -z "$DATASETS" || -z "$SEEDS" || -z "$APPROACHES" || -z "$CPU_CORES" ]]; then
    usage
fi

# Set flag variable for --is-flat
if [ "$IS_FLAT" -eq 1 ]; then
    FLAT_FLAG=" --is-flat"
else
    FLAT_FLAG=""
fi

# Split comma-separated lists into arrays
IFS=',' read -ra DATASET_ARR <<< "$DATASETS"
IFS=',' read -ra SEED_ARR <<< "$SEEDS"
IFS=',' read -ra APPROACH_ARR <<< "$APPROACHES"

# Create an array to store commands
commands=()

# Create experiments for each combination
for dataset in "${DATASET_ARR[@]}"; do
    for approach in "${APPROACH_ARR[@]}"; do
        for seed in "${SEED_ARR[@]}"; do
            # Compose log_dir: e.g. ../results_{dataset}_{approach}_6f_20p
            LOG_DIR="../results_${dataset}_${approach}_6f_20p"
            mkdir -p "${LOG_DIR}"
            # Build the command:
            CMD="python main.py --dataset ${dataset} --approach ${approach} --seed ${seed} --log-dir ${LOG_DIR}${FLAT_FLAG}"
            CMD+=" ${EXTRA_ARGS[@]}"
            # Prepend a header with a separator and the exact command being executed.
            FULL_CMD="( echo '$(printf '+%.0s' {1..100})'; echo 'Command: ${CMD}'; ${CMD} ) > ${LOG_DIR}/output.log 2>&1"
            commands+=("$FULL_CMD")
        done
    done
done

TOTAL=${#commands[@]}
echo "Total experiments: $TOTAL"

# Run experiments in parallel with progress reporting if GNU parallel is available.
if command -v parallel > /dev/null 2>&1; then
    printf "%s\n" "${commands[@]}" | parallel --progress --joblog ../log_dump/joblog.txt -j "$CPU_CORES"
else
    echo "GNU parallel not found; using xargs without progress tracking."
    printf "%s\n" "${commands[@]}" | xargs -I CMD -P "$CPU_CORES" bash -c CMD
fi

echo "All experiments completed."
