#!/bin/bash

for i in {15..25..5}; do
    echo "Starting optimization for hopper_${i}.yml..."
    python nn_sweep_k.py config/env/hopper_${i}.yml config/policy/hopper/nn_lwr.yml
    echo "Finished optimization for hopper_${i}.yml."
done

echo "All optimizations completed!"
