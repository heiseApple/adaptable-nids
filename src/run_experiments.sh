#!/bin/bash
# run_experiments.sh
# This script runs experiments in a combinatorial fashion.
# It accepts:
#   --src-dataset (-sd) : a comma-separated list of source datasets
#   --trg-dataset (-td) : a comma-separated list of target datasets
#   --seed (-s)         : a comma-separated list of seeds
#   --approach (-a)     : a comma-separated list of approaches
#   --is-flat (-f)      : a flag indicating flat structure (if present, add to command)
#   --cpu (-c)          : an integer for the number of cores to use
#   --extra-args (-e)   : extra arguments to pass directly to "python main.py"

usage() {
    echo "Usage: $0 --src-dataset <sd1,sd2,...> --trg-dataset <td1,td2,...> --seed <s1,s2,...> --approach <appr1,appr2,...> [--is-flat] --cpu <num_cores> [--extra-args \"<args>\"]"
    exit 1
}

# Default values
SRC_DATASETS=""
TRG_DATASETS=""
SEEDS=""
APPROACHES=""
IS_FLAT=0
CPU_CORES=""
EXTRA_ARGS=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -sd|--src-dataset)
            SRC_DATASETS="$2"
            shift ;;
        -td|--trg-dataset)
            TRG_DATASETS="$2"
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
            EXTRA_ARGS="$2"
            shift ;;
        *)
            echo "Unknown parameter passed: $1"
            usage ;;
    esac
    shift
done

if [[ -z "$SRC_DATASETS" || -z "$TRG_DATASETS" || -z "$SEEDS" || -z "$APPROACHES" || -z "$CPU_CORES" ]]; then
    usage
fi

# Set flag variable for --is-flat
if [ "$IS_FLAT" -eq 1 ]; then
    FLAT_FLAG=" --is-flat"
else
    FLAT_FLAG=""
fi

# Split comma-separated lists into arrays
IFS=',' read -ra SRC_DATASET_ARR <<< "$SRC_DATASETS"
IFS=',' read -ra TRG_DATASET_ARR <<< "$TRG_DATASETS"
IFS=',' read -ra SEED_ARR <<< "$SEEDS"
IFS=',' read -ra APPROACH_ARR <<< "$APPROACHES"

# Create an array to store commands
commands=()

# Create experiments for each combination
for src_dataset in "${SRC_DATASET_ARR[@]}"; do
    for trg_dataset in "${TRG_DATASET_ARR[@]}"; do
        # Skip iteration if src_dataset == trg_dataset
        if [[ "$src_dataset" == "$trg_dataset" ]]; then
            continue
        fi
        for approach in "${APPROACH_ARR[@]}"; do
            for seed in "${SEED_ARR[@]}"; do
                # Compose log_dir: e.g. ../results_{src_dataset}_{trg_dataset}_{approach}_6f_20p
                LOG_DIR="../results_sd_${src_dataset}_td_${trg_dataset}_${approach}_6f_20p"
                mkdir -p "${LOG_DIR}"
                # Build the command for execution
                CMD="python main.py --src-dataset ${src_dataset} --trg-dataset ${trg_dataset} --approach ${approach} --seed ${seed} --log-dir ${LOG_DIR}${FLAT_FLAG} ${EXTRA_ARGS}"
                # Prepend a header with a separator and the exact command being executed.
                FULL_CMD="( echo '$(printf '=%.0s' {1..100})'; echo 'Command: ${CMD}'; ${CMD} ) > ${LOG_DIR}/output.log 2>&1"
                commands+=("$FULL_CMD")
            done
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
