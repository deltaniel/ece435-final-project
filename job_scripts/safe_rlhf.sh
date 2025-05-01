#!/bin/bash

set -e

JOB_NAME="safe_rlhf"
RUN_TIME="00:10:00"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="output"
BATCH_SIZE=32
NUM_EPOCHS=1

LOG_PATH="${OUTPUT_DIR}/slurm-%j.out"

while getopts "o:b:n:j:t:" opt; do
  case $opt in
    o) OUTPUT_DIR="$OPTARG" ;;
    b) BATCH_SIZE="$OPTARG" ;;
    n) NUM_EPOCHS="$OPTARG" ;;
    j) JOB_NAME="$OPTARG" ;;
    t) RUN_TIME="$OPTARG" ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 1 ;;
  esac
done

# Log settings
echo "Submitting SLURM job with:"
echo "  JOB_NAME=${JOB_NAME}"
echo "  RUN_TIME=${RUN_TIME}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  BATCH_SIZE=${BATCH_SIZE}"
echo "  NUM_EPOCHS=${NUM_EPOCHS}"

mkdir -p "$OUTPUT_DIR"

sbatch --job-name="$JOB_NAME" \
       --time="$RUN_TIME" \
       --output="$LOG_PATH" \
       --export=ALL,OUTPUT_DIR="$OUTPUT_DIR",BATCH_SIZE="$BATCH_SIZE",NUM_EPOCHS="$NUM_EPOCHS" \
       "${SCRIPT_DIR}/safe_rlhf.slurm"
