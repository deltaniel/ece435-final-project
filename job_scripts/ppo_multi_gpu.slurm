#!/bin/bash
#SBATCH --job-name=train_ppo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=daniel.ruan@princeton.edu

module purge
module load cudatoolkit/12.6
module load anaconda3/2024.10
conda activate safe_rlhf

cd src/ece435_final_project
accelerate launch --multi_gpu --mixed_precision bf16 ppo_vanilla.py
