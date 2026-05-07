import argparse
import os
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Submit a job to SLURM or run locally.")
    parser.add_argument("command", nargs="+", help="Command to run")
    parser.add_argument("--job-name", default="darp_job", help="SLURM job name")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--ntasks-per-node", type=int, default=1, help="Tasks per node (usually GPUs)")
    parser.add_argument("--cpus-per-task", type=int, default=8, help="CPUs per task")
    parser.add_argument("--mem", default="32G", help="Memory")
    parser.add_argument("--time", default="24:00:00", help="Time limit")
    parser.add_argument("--partition", help="SLURM partition")
    parser.add_argument("--account", help="SLURM account")
    parser.add_argument("--local", action="store_true", help="Run locally instead of submitting to SLURM")
    
    args = parser.parse_args()

    full_command = " ".join(args.command)

    if args.local:
        print(f"Running locally: {full_command}")
        env = os.environ.copy()
        env["MUJOCO_GL"] = "egl"
        env["PYOPENGL_PLATFORM"] = "egl"
        subprocess.run(full_command, shell=True, env=env)
    else:
        # Construct sbatch command
        sbatch_cmd = [
            "sbatch",
            f"--job-name={args.job_name}",
            f"--nodes={args.nodes}",
            f"--ntasks-per-node={args.ntasks_per_node}",
            f"--cpus-per-task={args.cpus_per_task}",
            f"--mem={args.mem}",
            f"--time={args.time}",
        ]
        if args.partition:
            sbatch_cmd.append(f"--partition={args.partition}")
        if args.account:
            sbatch_cmd.append(f"--account={args.account}")

        # Use the template
        template_path = "scripts/slurm_template.slurm"
        if not os.path.exists(template_path):
            print(f"Error: Template {template_path} not found.")
            sys.exit(1)

        sbatch_cmd.append(template_path)
        sbatch_cmd.extend(args.command)

        print(f"Submitting to SLURM: {' '.join(sbatch_cmd)}")
        subprocess.run(sbatch_cmd)

if __name__ == "__main__":
    main()
