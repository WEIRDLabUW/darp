#!/bin/bash

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
