#!/bin/bash

# RESULTS CONSISTENT ON l40/l40s with 2 GPUS
# Verify MuJoCo Scores
task="hopper"

# CONFIRMED 2 GPU
env=config/env/$task/base.yml
policy=config/policy/$task/bc.yml
python launch_train.py $env $policy --fast
python launch_eval.py $env $policy --trials 100 --batched

# CONFIRMED 2 GPU
policy=config/policy/${task}/darp.yml
python launch_train.py $env $policy --fast
python launch_eval.py $env $policy --trials 100 --batched

task="ant"

# CONFIRMED 2 GPU
env=config/env/$task/base.yml
policy=config/policy/${task}/bc.yml
python launch_train.py $env $policy --fast
python launch_eval.py $env $policy --trials 100 --batched

# CONFIRMED 2 GPU
policy=config/policy/${task}/darp.yml
python launch_train.py $env $policy --fast
python launch_eval.py $env $policy --trials 100 --batched

task="walker"

# CONFIRMED 2 GPU
env=config/env/$task/base.yml
policy=config/policy/${task}/bc.yml
python launch_train.py $env $policy --fast
python launch_eval.py $env $policy --trials 100 --batched

# CONFIRMED 2 GPU
policy=config/policy/${task}/darp.yml
python launch_train.py $env $policy --fast
python launch_eval.py $env $policy --trials 100 --batched

task="halfcheetah"

# CONFIRMED 2 GPU
env=config/env/$task/base.yml
policy=config/policy/${task}/bc.yml
python launch_train.py $env $policy --fast
python launch_eval.py $env $policy --trials 100 --batched

# CONFIRMED 2 GPU
policy=config/policy/${task}/darp.yml
python launch_train.py $env $policy --fast
python launch_eval.py $env $policy --trials 100 --batched