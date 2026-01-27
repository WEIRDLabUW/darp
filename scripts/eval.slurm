#!/bin/bash

#SBATCH --job-name=hopper_test
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-a40

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH -o raw_log/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=quinnpfeifer@icloud.com
#SBATCH --requeue

if [ -z "$SLURM_JOB_ID" ]; then
    echo "Error: SLURM_JOB_ID is not set. Make sure you're not in the login node."
    exit 1
fi

echo "NAME: ${SLURM_JOB_NAME}, ID: ${SLURM_JOB_ID}"
export | grep SLURM

expected_gpus=$SLURM_NTASKS
actual_gpus=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')
echo $expected_gpus
echo $actual_gpus
if [[ $actual_gpus -ne $expected_gpus ]]; then
  echo "GPU ERROR: Allocation mismatch!"
  echo "REQUESTED: $expected_gpus GPUs"
  echo "ALLOCATED: $actual_gpus GPUs ($CUDA_VISIBLE_DEVICES)"
  rm $SLURM_SUBMIT_DIR/raw_log/${SLURM_JOB_NAME}_$SLURM_JOB_ID.log
  echo "Exiting job."
  exit 1
fi

nvidia-smi
nvidia-smi -q

bad_gpus=$(nvidia-smi -q | grep "GPU requires reset")

if [[ $bad_gpus ]]; then
  echo "Found GPUs that need reset!"
  rm $SLURM_SUBMIT_DIR/raw_log/${SLURM_JOB_NAME}_$SLURM_JOB_ID.log
  echo "Exiting job."
  exit 1
fi

source ~/.bashrc
conda activate /gscratch/weirdlab/quinn/nearest-sequence/envs/robosuite
#conda activate /gscratch/weirdlab/quinn/nearest-sequence/envs/robocasa

cd /gscratch/weirdlab/quinn/darp
#conda activate nearest-sequence

# Force software rendering
#export MUJOCO_GL=osmesa
#export PYOPENGL_PLATFORM=osmesa
#unset EGL_DEVICE_ID
#export DISPLAY="" 
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=device
export EGL_DEVICE_ID=0
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export GPUS=$CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_HOME=/mmfs1/sw/cuda/12.3.2
export NCCL_DEBUG=INFO
# export TORCH_LOGS="+dynamo"
# export TORCHDYNAMO_VERBOSE=1

# Set explicit paths for compilation
export CPATH=$CONDA_PREFIX/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/gscratch/weirdlab/quinn/nearest-sequence/video_action_learning/
export PYTHONPATH=$PYTHONPATH:/gscratch/weirdlab/quinn/nearest-sequence/robotwin/
#export PYTHONPATH=$PYTHONPATH:/gscratch/weirdlab/quinn/nearest-sequence/PyTorch-VAE/
#export PYTHONPATH=$PYTHONPATH:/gscratch/weirdlab/quinn/nearest-sequence/vaes/
#export LD_PRELOAD=$CONDA_PREFIX/lib/libGLEW.so
#export LD_PRELOAD=$CONDA_PREFIX/lib/libEGL.so:$LD_PRELOAD
#export LD_PRELOAD=$CONDA_PREFIX/lib/libGL.so:$LD_PRELOAD
# For Mujoco
unset LD_PRELOAD
#export LD_PRELOAD=$CONDA_PREFIX/lib/libGL.so
export TORCH_DISTRIBUTED_DEBUG=DETAIL
#export NCCL_DEBUG=WARN

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export VK_ICD_FILENAMES="$CONDA_PREFIX/share/vulkan/icd.d/lvp_icd.x86_64.json:/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json"

rm data/*lookup*
rm data/*standardized*
rm data/*/*standardized*
# rm -rf tensorboard_logs/*
#find ./log/ -mtime +7 -exec rm {} \;
#find ./raw_log/ -mtime +7 -exec rm {} \;
cp $0 $SLURM_SUBMIT_DIR/launch_files/${SLURM_JOB_NAME}_$SLURM_JOB_ID.slurm

set -e

run="eval export PROFILE=0; python -u"
#run="eval export PROFILE=1; kernprof -l -v"

#task="three_piece_assembly_task_D0"
#task="threading_task_D0"
#task="stack_task_D0"
# task="stack_three_task_D0"
#task="square_task_D0"
#task="hopper"
#task="walker"
#task="ant"
#task="maze"
#task="halfcheetah"
#task="push_t"
#task="pick_diverse_bottles"
#task="click_bell"
#task="close_drawer"
#task="close_single_door"
#task="turn_on_stove"
#task="turn_off_stove"
#task="counter_to_microwave"
#task="counter_to_sink"
data_size=100

#env=config/env/$task/base_val.yml
#env=config/env/$task/base_render.yml
#env=config/env/$task/base_diffusion.yml
#env=config/env/$task/base_diffusion_debug.yml
#env=config/env/$task/base_human.yml
#env=config/env/$task/base.yml
#env=config/env/$task/overfit.yml
#env=config/env/$task/base_1000.yml
#env=config/env/$task/low_dim.yml
#env=config/env/$task/r3m.yml
#env=config/env/$task/r3m_mixed.yml

#policy=config/policy/dan_base_model.yml
#policy=config/policy/dan_diffusion.yml
#policy=config/policy/diffusion.yml
#policy=config/policy/diffusion_seq_unet.yml
#policy=config/policy/diffusion_seq.yml
#policy=config/policy/${task}_low_dim/gmm.yml
# policy=config/policy/gmm.yml
# policy=config/policy/dan_no_delta.yml
# policy=config/policy/retrieve_and_play.yml
# policy=config/policy/$task/dan_no_action.yml
# policy=config/policy/$task/dan_no_delta.yml
# policy=config/policy/dan_no_action.yml
#policy=config/policy/hopper/dan_bc_arch_sanity.yml
#policy=config/policy/hopper/dan_scalar.yml
#policy=config/policy/hopper/dan_scalar_norm.yml
#policy=config/policy/hopper/dan_weighted_sum.yml
#policy=config/policy/$task/dan_bc_arch.yml
# policy=config/policy/${task}_low_dim/dan_no_delta_16.yml
#policy=config/policy/push_t_dan_model.yml
# policy=config/policy/dan_cosine.yml
#policy=config/policy/dan_val.yml
# policy=config/policy/dan_random_neighbors.yml
# policy=config/policy/dan_delta_magnitude.yml
# policy=config/policy/dan_permutation_dependent.yml
#policy=config/policy/dan_set_transformer.yml
# policy=config/policy/dan_gmm.yml
#policy=config/policy/dan_set_transformer_gmm.yml
# policy=config/policy/dan_set_transformer_2_model.yml 
#policy=config/policy/dan_deep_set.yml
#policy=config/policy/dan_deep_set_gmm.yml
#policy=config/policy/bc_r3m.yml
#policy=config/policy/${task}_low_dim/bc.yml
#policy=config/policy/dan_regularizer.yml
#policy=config/policy/${task}_low_dim/dan.yml
#policy=config/policy/${task}_low_dim/dan_diffusion_debug.yml
policy=config/policy/${task}_low_dim/dan_diffusion.yml
#policy=config/policy/${task}_low_dim/diffusion.yml
#policy=config/policy/${task}_low_dim/dan_val.yml
# policy=config/policy/${task}_low_dim/dan_nn.yml
# policy=config/policy/${task}_low_dim/dan_monolith.yml
#policy=config/policy/${task}_low_dim/dan_monolith_no_delta.yml
# policy=config/policy/${task}_low_dim/dan_monolithic.yml
# policy=config/policy/${task}/dan_monolithic.yml
#policy=config/policy/${task}/dan.yml
#policy=config/policy/${task}/bc.yml
#policy=config/policy/bc.yml
# policy=config/policy/bc_ensemble.yml
#policy=config/policy/lwr.yml
# policy=config/policy/lwr_r3m.yml
# policy=config/policy/dan_r3m.yml
#policy=config/policy/dan_r3m_cosine.yml
# policy=config/policy/hopper/regent_debug.yml
#policy=config/policy/hopper/regent.yml
#policy=config/policy/hopper/dan.yml
#policy=config/policy/hopper/dan_val.yml

# Various data processing scripts
# $run robotwin/script/collect_data.py $task demo_clean
#$run hdf5_to_d4rl.py data/$task/demo_human.hdf5 --goal
#$run hdf5_to_d4rl.py data/$task/demo.hdf5 --goal
#$run robotwin_to_pkl.py robotwin/data/$task/demo_randomized/data/ data/$task/
#$run robotwin_to_pkl.py robotwin/data/$task/demo_clean/data/ data/$task/
# $run pkl_to_actions.py data/$task/${data_size}_rgb.pkl
# mv data/$task/${data_size}_rgb_actions.pkl data/$task/${data_size}_actions.pkl
#$run robosuite_playground.py $env
#$run robosuite_state_to_rgb.py $env
#$run robosuite_state_to_rgb.py $env --high-quality
# $run robosuite_traj_viz.py $env
#$run robosuite_state_to_low_dim.py $env
# $run rgb_to_r3m.py data/$task/${data_size}_proprio.pkl data/$task/${data_size}_rgb.pkl data/$task/${data_size}_r3m.pkl --prop
#python -m torch.distributed.launch --nproc_per_node=2 --master_port=29501 robotwin_test.py

# Train
#$run launch_train.py $env $policy
#$run launch_train.py $env $policy --sloppy
#$run launch_train.py $env $policy --eval-epochs 10 --eval-trials 1
#$run launch_train.py $env $policy --eval-epochs 1 --eval-trials 1
#$run launch_train.py $env $policy --eval-epochs 5 --eval-trials 50 --sloppy
#$run launch_train.py $env $policy --eval-epochs 20 --eval-trials 100 --sloppy

POLICIES=(
    "config/policy/${task}_low_dim/dan_128.yml"
    "config/policy/${task}_low_dim/dan_64.yml"
    "config/policy/${task}_low_dim/dan_32.yml"
    "config/policy/${task}_low_dim/dan_16.yml"
    "config/policy/${task}_low_dim/bc_128.yml"
    "config/policy/${task}_low_dim/bc_64.yml"
    "config/policy/${task}_low_dim/bc_32.yml"
    "config/policy/${task}_low_dim/bc_16.yml"
    "config/policy/${task}_low_dim/dan_no_delta_128.yml"
    "config/policy/${task}_low_dim/dan_no_delta_64.yml"
    "config/policy/${task}_low_dim/dan_no_delta_32.yml"
    "config/policy/${task}_low_dim/dan_no_delta_16.yml"
)
#policy=${POLICIES[$SLURM_ARRAY_TASK_ID]}

#$run launch_train.py $env $policy --eval-epochs 5 --sloppy --eval-trials 100

#$run launch_train.py $env $policy --eval-epochs 10 --eval-trials 100 --sloppy --start-eval-epoch 150
#$run train_bc.py $env $policy --sloppy

# Eval
#$run launch_eval.py $env $policy --trials 100 --batched
#$run launch_eval.py $env $policy --trials 1
#$run eval_model.py $env $policy --trials 1 --batched
#$run eval_model.py $env $policy --trials 1
#$run launch_eval.py $env $policy --trials 100 --batched --trials_per_worker 2
#$run launch_eval.py $env $policy --trials 100

# Run optimizer
# $run delete_study.py $SLURM_JOB_NAME
#$run launch_optimize.py 1 $env $policy $SLURM_JOB_NAME
#$run launch_optimize.py 2 $env $policy $SLURM_JOB_NAME
#$run launch_optimize.py 3 $env $policy $SLURM_JOB_NAME
# $run launch_sweep.py $env $policy $SLURM_JOB_NAME

# RESULTS CONSISTENT ON l40/l40s with 2 GPUS
# Verify MuJoCo Scores
task="hopper"

# CONFIRMED 2 GPU
env=config/env/$task/base.yml
policy=config/policy/$task/bc.yml
$run launch_train.py $env $policy --sloppy
$run launch_eval.py $env $policy --trials 100 --batched

# CONFIRMED 2 GPU
policy=config/policy/${task}/darp.yml
$run launch_train.py $env $policy --sloppy
$run launch_eval.py $env $policy --trials 100 --batched

task="ant"

# CONFIRMED 2 GPU
env=config/env/$task/base.yml
policy=config/policy/${task}/bc.yml
$run launch_train.py $env $policy --sloppy
$run launch_eval.py $env $policy --trials 100 --batched

# CONFIRMED 2 GPU
policy=config/policy/${task}/darp.yml
$run launch_train.py $env $policy --sloppy
$run launch_eval.py $env $policy --trials 100 --batched

task="walker"

# CONFIRMED 2 GPU
env=config/env/$task/base.yml
policy=config/policy/${task}/bc.yml
$run launch_train.py $env $policy --sloppy
$run launch_eval.py $env $policy --trials 100 --batched

# CONFIRMED 2 GPU
policy=config/policy/${task}/darp.yml
$run launch_train.py $env $policy --sloppy
$run launch_eval.py $env $policy --trials 100 --batched

task="halfcheetah"

# CONFIRMED 2 GPU
env=config/env/$task/base.yml
policy=config/policy/${task}/bc.yml
$run launch_train.py $env $policy --sloppy
$run launch_eval.py $env $policy --trials 100 --batched

# CONFIRMED 2 GPU
policy=config/policy/${task}/darp.yml
$run launch_train.py $env $policy --sloppy
$run launch_eval.py $env $policy --trials 100 --batched

task="stack_task_D0"

# CONFIRMED 2 GPU
env=config/env/$task/base.yml
policy=config/policy/${task}_low_dim/bc.yml
$run launch_train.py $env $policy --sloppy
$run launch_eval.py $env $policy --trials 100 --batched

# CONFIRMED 2 GPU
policy=config/policy/${task}_low_dim/darp.yml
$run launch_train.py $env $policy --sloppy
$run launch_eval.py $env $policy --trials 100 --batched

# CONFIRMED 4 GPU
env=config/env/$task/r3m.yml
policy=config/policy/${task}_r3m/bc_r3m.yml
$run launch_train.py $env $policy --sloppy
$run launch_eval.py $env $policy --trials 100 --batched

# CONFIRMED 4 GPU
policy=config/policy/${task}_r3m/darp_r3m.yml
$run launch_train.py $env $policy --sloppy
$run launch_eval.py $env $policy --trials 100 --batched

task="threading_task_D0"

# CONFIRMED 2 GPU
env=config/env/$task/base.yml
policy=config/policy/${task}_low_dim/bc.yml
$run launch_train.py $env $policy --sloppy
$run launch_eval.py $env $policy --trials 100 --batched

# CONFIRMED 2 GPU
policy=config/policy/${task}_low_dim/darp.yml
$run launch_train.py $env $policy --sloppy
$run launch_eval.py $env $policy --trials 100 --batched

env=config/env/$task/r3m.yml
policy=config/policy/${task}_r3m/bc_r3m.yml
$run launch_train.py $env $policy --sloppy
$run launch_eval.py $env $policy --trials 100 --batched

# CONFIRMED 5 GPU
policy=config/policy/${task}_r3m/darp_r3m.yml
$run launch_train.py $env $policy --sloppy
$run launch_eval.py $env $policy --trials 100 --batched

task="square_task_D0"

# CONFIRMED 2 GPU
env=config/env/$task/base.yml
policy=config/policy/${task}_low_dim/bc.yml
$run launch_train.py $env $policy --sloppy
$run launch_eval.py $env $policy --trials 100 --batched

# CONFIRMED 2 GPU
policy=config/policy/${task}_low_dim/darp.yml
$run launch_train.py $env $policy --sloppy
$run launch_eval.py $env $policy --trials 100 --batched

# CONFIRMED 4 GPU
env=config/env/$task/r3m.yml
policy=config/policy/${task}_r3m/bc_r3m.yml
$run launch_train.py $env $policy --sloppy
$run launch_eval.py $env $policy --trials 100 --batched

# CONFIRMED 4 GPU
policy=config/policy/${task}_r3m/darp_r3m.yml
$run launch_train.py $env $policy --sloppy
$run launch_eval.py $env $policy --trials 100 --batched

task="push_t"
env=config/env/$task/base.yml
policy=config/policy/${task}/bc.yml
$run launch_train.py $env $policy --sloppy
$run launch_eval.py $env $policy --trials 100 --batched

policy=config/policy/${task}/darp.yml
$run launch_train.py $env $policy --sloppy
$run launch_eval.py $env $policy --trials 100 --batched

task="close_drawer"

env=config/env/$task/base.yml
policy=config/policy/${task}/bc.yml
$run launch_train.py $env $policy --sloppy
$run launch_eval.py $env $policy --trials 100 --batched

policy=config/policy/${task}/darp.yml
$run launch_train.py $env $policy --sloppy
$run launch_eval.py $env $policy --trials 100 --batched

task="close_single_door"

env=config/env/$task/base.yml
policy=config/policy/${task}/bc.yml
$run launch_train.py $env $policy --sloppy
$run launch_eval.py $env $policy --trials 100 --batched

policy=config/policy/${task}/darp.yml
$run launch_train.py $env $policy --sloppy
$run launch_eval.py $env $policy --trials 100 --batched

task="turn_on_stove"

env=config/env/$task/base.yml
policy=config/policy/${task}/bc.yml
$run launch_train.py $env $policy --sloppy
$run launch_eval.py $env $policy --trials 100 --batched

policy=config/policy/${task}/darp.yml
$run launch_train.py $env $policy --sloppy
$run launch_eval.py $env $policy --trials 100 --batched
