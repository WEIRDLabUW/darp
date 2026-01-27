#!/bin/bash

#SBATCH --job-name=knn
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40s

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=2
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH -o log/%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=quinnpfeifer@icloud.com

echo "ID: ${SLURM_JOB_ID}"
source ~/.bashrc
#cd /gscratch/weirdlab/quinn/
cd /gscratch/weirdlab/quinn/nearest-sequence/data/

du -BG --max-depth=3 . | sort -n
