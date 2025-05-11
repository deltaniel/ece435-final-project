#!/bin/bash
#SBATCH --job-name=train_ppo_test
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

cd src/ece435_final_project
python ppo_vanilla.py
