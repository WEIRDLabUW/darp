import os
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
import sys
import subprocess

import torch

def main():
    nproc_per_node = torch.cuda.device_count() or 1

    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc_per_node}",
        "train.py"
    ] + sys.argv[1:]

    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
