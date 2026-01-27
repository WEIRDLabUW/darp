#!/bin/bash

#SBATCH --job-name=knn
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-a40

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH -o log/%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=quinnpfeifer@icloud.com

echo "ID: ${SLURM_JOB_ID}"
source ~/.bashrc
cd /gscratch/weirdlab/quinn/nearest-sequence
conda activate nearest-sequence

# Force software rendering
#export MUJOCO_GL=osmesa
#export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
echo "SLURM GPUS: ${CUDA_VISIBLE_DEVICES}"
export GPUS=$CUDA_VISIBLE_DEVICES
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_HOME=/mmfs1/sw/cuda/12.3.2

# Set explicit paths for compilation
export CPATH=$CONDA_PREFIX/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=:/gscratch/weirdlab/quinn/nearest-sequence/video_action_learning/
#export LD_PRELOAD=$CONDA_PREFIX/lib/libGLEW.so
#export LD_PRELOAD=$CONDA_PREFIX/lib/libEGL.so:$LD_PRELOAD
#export LD_PRELOAD=$CONDA_PREFIX/lib/libGL.so:$LD_PRELOAD
# For Mujoco
unset LD_PRELOAD
#export LD_PRELOAD=$CONDA_PREFIX/lib/libGLEW.so:$CONDA_PREFIX/lib/libGL.so
#export LD_PRELOAD=$CONDA_PREFIX/lib/libGLEW.so
#export LD_PRELOAD=$CONDA_PREFIX/lib/libEGL.so:$LD_PRELOAD
export CUDA_LAUNCH_BLOCKING=1
rm data/*lookup*

run_on_gpu() {
    local i=$1
    local gpu_id=$2

    export CUDA_VISIBLE_DEVICES=$gpu_id

    python -u nn_eval.py config/env/stack_task_D0_img.yml config/policy/bc.yml $i

    echo "Task $i completed on GPU $gpu_id"
}

num_gpus=4

for i in {1..20}; do
    gpu_id=$((($i-1) % num_gpus))

    run_on_gpu $i $gpu_id &

    sleep 1
done

wait

sum=0
count=0
for result_file in results/*.txt; do
    value=$(cat "$result_file")
    sum=$(echo "$sum + $value" | bc -l)
    count=$((count + 1))
done

mean=$(echo "scale=4; $sum / $count" | bc -l)
echo "Mean of all results: $mean"
