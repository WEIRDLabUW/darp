#!/bin/bash
#SBATCH --job-name=knn
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -o log/%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=quinnpfeifer@icloud.com

echo "ID: ${SLURM_JOB_ID}"
source ~/.bashrc
cd /gscratch/weirdlab/quinn/nearest-sequence
conda activate nearest-sequence

# Clear any existing GL/EGL configs
unset LD_PRELOAD
unset PYOPENGL_PLATFORM
unset MUJOCO_GL
unset DISPLAY

# Set specific configuration for mujoco-py
export MUJOCO_GL=osmesa
export PYTHONPATH=$CONDA_PREFIX/lib/python3.10/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Keep CUDA configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Debug info
echo "Environment Setup:"
echo "MUJOCO_GL=$MUJOCO_GL"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# Try to create a test render before running main script
python - << EOF
import mujoco_py
import gym
env = gym.make('Hopper-v2')
viewer = mujoco_py.MjRenderContextOffscreen(env.sim, device_id=-1)
frame = viewer.read_pixels(512, 512, depth=False)
print("Test render successful")
EOF

# If test passes, run main script
python -u state_to_img.py config/env/hopper.yml
