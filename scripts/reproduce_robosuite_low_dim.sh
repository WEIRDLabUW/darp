#!/bin/bash

# Robosuite Low-Dim Reproduction

task="stack_task_D0"

# BC
env=config/env/$task/base.yml
policy=config/policy/${task}_low_dim/bc.yml
python launch_train.py $env $policy --fast
python launch_eval.py $env $policy --trials 100 --batched

# DARP
policy=config/policy/${task}_low_dim/darp.yml
python launch_train.py $env $policy --fast
python launch_eval.py $env $policy --trials 100 --batched

task="square_task_D0"

# BC
env=config/env/$task/base.yml
policy=config/policy/${task}_low_dim/bc.yml
python launch_train.py $env $policy --fast
python launch_eval.py $env $policy --trials 100 --batched

# DARP
policy=config/policy/${task}_low_dim/darp.yml
python launch_train.py $env $policy --fast
python launch_eval.py $env $policy --trials 100 --batched

task="threading_task_D0"

# BC
env=config/env/$task/base.yml
policy=config/policy/${task}_low_dim/bc.yml
python launch_train.py $env $policy --fast
python launch_eval.py $env $policy --trials 100 --batched

# DARP
policy=config/policy/${task}_low_dim/darp.yml
python launch_train.py $env $policy --fast
python launch_eval.py $env $policy --trials 100 --batched
