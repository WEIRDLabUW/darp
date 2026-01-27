import os
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
import gym
gym.logger.set_level(40)
import yaml
from argparse import ArgumentParser
from eval import batched_eval, parallel_eval
from train import train_model

from logging_util import logger

def main():
    parser = ArgumentParser()
    parser.add_argument("env_config_path", help="Path to environment config file")
    parser.add_argument("policy_config_path", help="Path to policy config file")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--trials_per_worker", type=int, default=1)
    parser.add_argument("--results_file_name", default=None)
    parser.add_argument("--batched", action="store_true")
    args, _ = parser.parse_known_args()
    logger.info(f"Evaluating with {args.trials} trial{'s' if args.trials != 1 else ''}")

    with open(args.env_config_path, 'r') as f:
        env_cfg = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.policy_config_path, 'r') as f:
        policy_cfg = yaml.load(f, Loader=yaml.FullLoader)


    darp = policy_cfg['model_config'].get("darp", False)
    env_cfg['seed'] = 42
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    env_cfg['device'] = f"cuda:{local_rank}"
    policy_cfg['train_config']['force_retrain'] = False
    agent, _ = train_model(0, 1, env_cfg, policy_cfg)
    agent.eval()

    if args.batched:
        batched_eval(env_cfg, agent, trials=args.trials, results=args.results_file_name, reset=True, darp=darp, trials_per_worker=args.trials_per_worker)
    else:
        parallel_eval(env_cfg, agent, trials=args.trials, results=args.results_file_name, darp=darp)

if __name__ == "__main__":
    main()
