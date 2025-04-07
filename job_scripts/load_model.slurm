#!/bin/bash
#SBATCH --job-name=load-llama
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:01:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=dr5390@princeton.edu

module purge
module load anaconda3/2024.10
conda activate ece435

cd src/ece435_final_project
python pipeline.py
