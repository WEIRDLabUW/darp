import os
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
import gym
gym.logger.set_level(40)
import sys
import subprocess

import socket

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def main():
    eval_world_size = int(os.environ.get('SLURM_NTASKS', 1))
    free_port = find_free_port()

    cmd = [
        "python", "-m", "torch.distributed.launch",
        f"--nproc_per_node={eval_world_size}",
        f"--master_port={free_port}",
        "train.py"
    ] + sys.argv[1:]

    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
