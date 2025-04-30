#!/bin/bash
#SBATCH --job-name=safe_rlhf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --time=00:10:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=daniel.ruan@princeton.edu

module purge
module load cudatoolkit/12.6
module load anaconda3/2024.10
conda activate safe_rlhf

echo "Running job with:"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  BATCH_SIZE=${BATCH_SIZE}"
echo "  NUM_EPOCHS=${NUM_EPOCHS}"

cd src/ece435_final_project
python safe.rlhf.py -o "${OUTPUT_DIR}" -b "${BATCH_SIZE}" -n "${NUM_EPOCHS}"
