#!/bin/bash
#
#SBATCH --job-name=ppo_rlhf                # a short name for your job
#SBATCH --output=logs/ppo_rlhf_%j.out      # STDOUT → logs/ppo_rlhf_<jobid>.out
#SBATCH --error=logs/ppo_rlhf_%j.err       # STDERR → logs/ppo_rlhf_<jobid>.err
#SBATCH --time=04:00:00                    # walltime hh:mm:ss
#SBATCH --partition=standard               # or your cluster’s default CPU partition
#SBATCH --ntasks=1                         # one process
#SBATCH --gres=gpu:1                       # request GPU resources
#SBATCH --cpus-per-task=4                  # number of CPU cores
#SBATCH --mem=32G                          # total RAM
#SBATCH --mail-type=END,FAIL               # email notifications
#SBATCH --mail-user=pk7019@princeton.edu

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

module load anaconda3/2024.6
conda activate safe_rlhf

# ──────────────────────────────────────────────────────────────────────────────
# 2) Create a logs directory if it doesn’t exist
# ──────────────────────────────────────────────────────────────────────────────
mkdir -p logs

# ──────────────────────────────────────────────────────────────────────────────
# 3) Run your training
#    - your script writes DEBUG→ppo_rlhf.log, INFO→console/stdout
#    - slurm captures stdout/stderr in the files above
# ──────────────────────────────────────────────────────────────────────────────
python eval.py
# ──────────────────────────────────────────────────────────────────────────────
# 4) (Optional) at the end, echo where to find logs
# ──────────────────────────────────────────────────────────────────────────────
echo "Slurm stdout → ${SLURM_SUBMIT_DIR}/logs/ppo_rlhf_eval_${SLURM_JOB_ID}.out"
echo "Slurm stderr → ${SLURM_SUBMIT_DIR}/logs/ppo_rlhf_eval_${SLURM_JOB_ID}.err"
echo "Detailed logs → ${SLURM_SUBMIT_DIR}/ppo_rlhf_eval.log"