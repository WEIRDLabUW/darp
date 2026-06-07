import pickle
import os

from util import construct_env, reset_vision_ob, env_to_rgb_array, crop_obs_for_env
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
import numpy as np
import gym
gym.logger.set_level(40)
from argparse import ArgumentParser
import yaml

import cv2
from rgb_arrays_to_mp4 import rgb_arrays_to_mp4
import math

parser = ArgumentParser()
parser.add_argument("env_config_path", help="Path to environment config file")
parser.add_argument("high_quality", action="store_true")

args, _ = parser.parse_known_args()

with open(args.env_config_path, 'r') as f:
    env_cfg = yaml.load(f, Loader=yaml.FullLoader)

data = pickle.load(open(env_cfg['demo_pkl'], 'rb'))

env = construct_env(env_cfg)
env_name = env_cfg['name']

camera_names = [env.env.sim.model.camera_id2name(i) for i in range(env.env.sim.model.ncam)]
print(f"Choose cameras from: {camera_names}")
camera_names = env_cfg['cams']
NUM_VIEWPOINTS = len(camera_names)

crops = env_cfg.get('crops', {})
print(crops)

if args.high_quality:
    height, width = 1024, 1024
else:
    height, width = 224, 224
print(height)
grid_side_len = math.ceil(math.sqrt(len(data)))

img_data = []
proprio_data = []

def main():
    bad_trajs = []
    for traj in range(len(data)):
        frames = []
        print(f"Processing traj {traj}...")
        initial_state = dict(states=data[traj]['states'][0])
        initial_state["model"] = data[traj]["model_file"]
        traj_obs = []
        proprio_obs = []

        env.reset()
        env.reset_to(initial_state)
        reset_vision_ob()
        for ob in range(len(data[traj]['observations'])):
            full_frame = np.empty((height, 0, 3), dtype=np.uint8)
            for camera in camera_names:
                crop_corners = np.array(crops.get(camera, [[0., 0.], [1., 1.]]))
                frame = env_to_rgb_array(env, camera, np.copy(crop_corners), width, height)

                full_frame = np.hstack((full_frame, frame))

            split_frames = np.split(full_frame, NUM_VIEWPOINTS, axis=1)
            traj_obs.append(np.concatenate([frame.flatten() for frame in split_frames]))
            proprio_obs.append(crop_obs_for_env([env.get_observation()], env_name, env_instance=env, proprio=True)[0])
            full_frame = cv2.resize(full_frame, (height, width))
            frames.append(full_frame)

            if ob != len(data[traj]['observations']) - 1:
                next_state = dict(states=data[traj]['states'][ob + 1])
                env.reset_to(next_state)
            else:
                env.step(data[traj]['actions'][-1])

        img_data.append({'observations': np.array(traj_obs), 'actions': data[traj]['actions'], 'states': data[traj]['states']})
        proprio_data.append({'observations': np.array(proprio_obs), 'actions': data[traj]['actions'], 'states': data[traj]['states']})
        img_data[-1]['model_file'] = data[traj]['model_file']
        proprio_data[-1]['model_file'] = data[traj]['model_file']
        col, row = divmod(traj, grid_side_len)
        print(f"{col=}, {row=}")
        ep_length = len(data[traj]['observations'])
        frames = np.array(frames)

    print(f"Success! Dumping data to {env_cfg['demo_pkl'][:-4] + '_rgb.pkl'}")
    pickle.dump(img_data, open(env_cfg['demo_pkl'][:-4] + '_rgb.pkl', 'wb'))
    print(f"Success! Dumping data to {env_cfg['demo_pkl'][:-4] + '_proprio.pkl'}")
    pickle.dump(proprio_data, open(env_cfg['demo_pkl'][:-4] + '_proprio.pkl', 'wb'))
    print(f"Remove {bad_trajs}")
    good_expert_data = []
    for traj in range(len(data)):
        if traj not in bad_trajs:
            good_expert_data.append(data[traj])
    pickle.dump(good_expert_data, open(env_cfg['demo_pkl'], 'wb'))

main()
