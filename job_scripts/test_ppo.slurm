#!/bin/bash
#SBATCH --job-name=finetune-llama
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=00:01:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=na1225@princeton.edu

module purge
module load anaconda3/2024.10
conda activate safe_rlhf

cd src/ece435_final_project
python ppo_vanilla.py
