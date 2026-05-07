#!/bin/bash

#SBATCH --job-name=darp_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -o log/%j.out

# Source bashrc and activate conda environment if needed
# source ~/.bashrc
# conda activate darp

# Rendering settings
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}

# Run the command
echo "Running: $@"
$@
