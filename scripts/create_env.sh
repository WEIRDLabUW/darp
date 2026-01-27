#!/bin/bash

#SBATCH --job-name=knn
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40s

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=10:00:00
#SBATCH -o log/%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=quinnpfeifer@icloud.com

echo "ID: ${SLURM_JOB_ID}"
env_name="robosuite"
#env_name="robocasa"
source ~/.bashrc
cd /gscratch/weirdlab/quinn/nearest-sequence

#echo "PRE-DEACTIVATE: ${CONDA_DEFAULT_ENV}"
conda deactivate
#conda deactivate
#echo "PRE-REMOVE: ${CONDA_DEFAULT_ENV}"
conda env remove --prefix ./envs/$env_name
#conda clean -a
#echo "PRE-CREATE: ${CONDA_DEFAULT_ENV}"
conda env create --prefix ./envs/$env_name -f environment.yml
conda activate envs/$env_name

#rm -rf $CONDA_PREFIX/lib/python3.10/site-packages/robomimic*
#rm -rf $CONDA_PREFIX/lib/python3.10/site-packages/robosuite*
#rm -rf $CONDA_PREFIX/lib/python3.10/site-packages/mimicgen*
export CUDA_HOME=/mmfs1/sw/cuda/12.3.2/
#pip install --force-reinstall --no-cache-dir -e GroundingDINO

# Old robosuite
pip install -e ../cloned/mimicgen
pip install pynput==1.6.0
pip install -e ../cloned/robosuite
pip install -e ../cloned/robomimic
pip install mujoco==2.3.2
pip install numba==0.58.1

# New for casa
#pip install -e ../cloned/robocasa/robosuite_casa/robosuite/
#pip install -e ../cloned/robocasa/
#pip install -e ../cloned/robocasa/mimicgen_casa/
#pip install -e ../cloned/robocasa/robomimic_casa/

pip install optuna
pip install -e ../cloned/r3m
pip install line_profiler
pip install --upgrade protobuf

#git clone git@github.com:RoboTwin-Platform/RoboTwin.git
#export CUDA_HOME=/mmfs1/sw/cuda/12.3.2
#module load gcc/11.2.0
#./RoboTwin/script/_install.sh


ln -s $CONDA_PREFIX/lib/libEGL.so.1 $CONDA_PREFIX/lib/libEGL.so
mkdir -p $CONDA_PREFIX/include/X11
$CONDA_PREFIX/lib/python3.10/site-packages/egl_probe/glad/X11/*.h $CONDA_PREFIX/include/X11/
