#!/bin/bash

# Robosuite R3M Reproduction

task="square_task_D0"

# BC R3M
env=config/env/$task/r3m.yml
policy=config/policy/${task}_r3m/bc_r3m.yml
python launch_train.py $env $policy --fast
python launch_eval.py $env $policy --trials 100 --batched

# DARP R3M
policy=config/policy/${task}_r3m/darp_r3m.yml
python launch_train.py $env $policy --fast
python launch_eval.py $env $policy --trials 100 --batched


task="stack_task_D0"

# BC R3M
env=config/env/$task/r3m.yml
policy=config/policy/${task}_r3m/bc_r3m.yml
python launch_train.py $env $policy --fast
python launch_eval.py $env $policy --trials 100 --batched

# DARP R3M
policy=config/policy/${task}_r3m/darp_r3m.yml
python launch_train.py $env $policy --fast
python launch_eval.py $env $policy --trials 100 --batched


task="threading_task_D0"

# BC R3M
env=config/env/$task/r3m.yml
policy=config/policy/${task}_r3m/bc_r3m.yml
python launch_train.py $env $policy --fast
python launch_eval.py $env $policy --trials 100 --batched

# DARP R3M
policy=config/policy/${task}_r3m/darp_r3m.yml
python launch_train.py $env $policy --fast
python launch_eval.py $env $policy --trials 100 --batched
