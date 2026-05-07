import os
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
import gym
gym.logger.set_level(40)
import yaml
from argparse import ArgumentParser
import sys
import subprocess

import torch

def main():

    nproc_per_node = int(os.environ.get('SLURM_NTASKS', torch.cuda.device_count() or 1))

    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc_per_node}",
        "eval_model.py"
    ] + sys.argv[1:]

    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
