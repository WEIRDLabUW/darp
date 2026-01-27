#!/bin/bash

for i in {5..25..5}; do
    echo "Starting optimization for hopper_${i}.yml..."
    python nn_param_optimizer.py config/env/hopper_${i}.yml config/policy/hopper/ns_lwr.yml
    echo "Finished optimization for hopper_${i}.yml."
done

echo "All optimizations completed!"
