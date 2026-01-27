import pickle
import os

from nn_util import construct_env, reset_vision_ob, get_proprio, env_to_rgb_array, hide_robot, crop_obs_for_env
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
import numpy as np
import gym
gym.logger.set_level(40)
from argparse import ArgumentParser
import yaml

import cv2
from rgb_arrays_to_mp4 import rgb_arrays_to_mp4
import matplotlib.pyplot as plt
import math
import time

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
#camera_names = env_cfg['cams']
NUM_VIEWPOINTS = len(camera_names)

crops = env_cfg.get('crops', {})
#crops = {}
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
    #all_frames = np.zeros((env_cfg['ep_length'], grid_side_len * height, grid_side_len * width, 3))
    #cube_mask = np.ones((height, width))
    bad_trajs = []
    for traj in range(len(data)):
        frames = []
        print(f"Processing traj {traj}...")
        initial_state = dict(states=data[traj]['states'][0])
        initial_state["model"] = data[traj]["model_file"]
        traj_obs = []
        proprio_obs = []

        env.reset()
        breakpoint()
        env.reset_to(initial_state)
        #hide_robot(env.env.sim.model)
        reset_vision_ob()
        for ob in range(len(data[traj]['observations'])):
            #print(env.get_observation()['robot0_eef_pos'] - data[traj]['observations'][ob][:3])
            full_frame = np.empty((height, 0, 3), dtype=np.uint8)
            start = time.time()
            for camera in camera_names:
                crop_corners = np.array(crops.get(camera, [[0., 0.], [1., 1.]]))
                frame = env_to_rgb_array(env, camera, np.copy(crop_corners), width, height)

                full_frame = np.hstack((full_frame, frame))

            split_frames = np.split(full_frame, NUM_VIEWPOINTS, axis=1)
            traj_obs.append(np.concatenate([frame.flatten() for frame in split_frames]))
            proprio_obs.append(crop_obs_for_env([env.get_observation()], env_name, env_instance=env, proprio=True)[0])
            full_frame = cv2.resize(full_frame, (height, width))
            frames.append(full_frame)

            #plt.imsave("first_img.png", full_frame)
            #traj_obs.append(cv2.resize(full_frame, (height, width)).flatten())
            #env.step(data[traj]['actions'][ob])
            if ob != len(data[traj]['observations']) - 1:
                next_state = dict(states=data[traj]['states'][ob + 1])
                env.reset_to(next_state)
            else:
                env.step(data[traj]['actions'][-1])

        #print(env.is_success())
        #if env.get_reward() == 1:
        img_data.append({'observations': np.array(traj_obs), 'actions': data[traj]['actions'], 'states': data[traj]['states']})
        proprio_data.append({'observations': np.array(proprio_obs), 'actions': data[traj]['actions'], 'states': data[traj]['states']})
        img_data[-1]['model_file'] = data[traj]['model_file']
        proprio_data[-1]['model_file'] = data[traj]['model_file']
        #rgb_arrays_to_mp4(frames, f"data/{traj}.mp4")
        #else:
        #    print("REJECTING TRAJECTORY")
        #    bad_trajs.append(traj)
        #    rgb_arrays_to_mp4(frames, f"data/{traj}.mp4")
        col, row = divmod(traj, grid_side_len)
        print(f"{col=}, {row=}")
        ep_length = len(data[traj]['observations'])
        frames = np.array(frames)
        #rgb_arrays_to_mp4(frames, f"data/{traj}.mp4")
        # brightness = np.mean(frames, axis=3)
        # cube_mask *= np.prod(brightness >= 230, axis=0)
        # frames[:, cube_mask.astype(bool)] = 0
        #all_frames[:ep_length, col * height:(col + 1) * height, row * width:(row + 1) * width] = frames


    #rgb_arrays_to_mp4(all_frames, f"data/{env_name}.mp4")
    #y_bound, x_bound = np.where(~cube_mask.astype(bool))
    #
    # suggested_crop_corners = np.array([[x_bound.min(), y_bound.min()], [x_bound.max() + 1, y_bound.max() + 1]], dtype=np.float64)
    # suggested_crop_corners[:, 0] /= height
    # suggested_crop_corners[:, 1] /= width
    # crop_size = crop_corners[1] - crop_corners[0] 
    # crop_corners[1] = crop_corners[0] + suggested_crop_corners[1] * crop_size
    # crop_corners[0] += suggested_crop_corners[0] * crop_size
    #
    # print("[[{:.3f}, {:.3f}], [{:.3f}, {:.3f}]]".format(
    #     crop_corners[0][0], 
    #     crop_corners[0][1],
    #     crop_corners[1][0], 
    #     crop_corners[1][1]
    # ))

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
