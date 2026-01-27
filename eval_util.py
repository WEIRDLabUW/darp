import os

from constants import RESNET_SIZE
from push_t_env import PushTEnv
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'

import torch
import gym

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.utils.file_utils import get_env_metadata_from_dataset

import numpy as np

import mimicgen.utils.robomimic_utils as RobomimicUtils
# d4rl sets some logging setting, let's undo them
import logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
from logging_util import logger

import cv2

# To be populated if needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def construct_env(config, seed=None, gpu_id=0, lock=None, render=False):
    is_robosuite = config.get('robosuite', False)

    if is_robosuite:
        dummy_spec = dict(
            obs=dict(
                    low_dim=["robot0_eef_pos"],
                    rgb=[],
                ),
        )

        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

        env_meta = get_env_metadata_from_dataset(dataset_path=config['demo_hdf5'])
        env_meta['env_kwargs']['hard_reset'] = False
        env_meta['env_kwargs']['render_gpu_device_id'] = gpu_id
        env_meta['env_kwargs']['reward_shaping'] = config.get("reward_shaping", False)
        env_meta['env_kwargs']['has_offscreen_renderer'] = render
        if seed is not None:
            env_meta['seed'] = seed
        if robomimic.__version__ == "0.3.0":
            logger.debug(f"{gpu_id}:{seed} pre env from meta")
            env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render_offscreen=render, logger=logger, seed=seed, lock=lock)
            logger.debug(f"{gpu_id}:{seed} post env from meta")
        else:
            env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render_offscreen=render)

        camera_name = RobomimicUtils.get_default_env_cameras(env_meta=env_meta)[0]
        return env

    env_name = config['name']

    if env_name == 'push_t':
        env = PushTEnv()
    else:
        import d4rl
        env = gym.make(env_name)
        if env_name == "maze2d-umaze-v1":
            env = env.env
            env.reward_type = 'sparse'
    return env

def get_proprio(config, obs) -> np.ndarray:
    is_robosuite = config.get('robosuite', False)
    if is_robosuite:
        proprio_obs = np.array([])

        default_low_dim_obs = [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ]

        for key in default_low_dim_obs:
            proprio_obs = np.hstack((proprio_obs, obs[key]))

        return proprio_obs
    else:
        return obs

def get_processed_obs(observation, frame, env, model, config, obs_type):
    device = config['device']
    env_name = config.get('name', 10)

    if not isinstance(observation, list):
        observation = [observation]

    proprio_state = []
    for o in observation:
        if config.get('add_proprio', False):
            proprio_state.append(get_proprio(config, o))
        else:
            proprio_state.append(np.array([]))

    proprio_state = np.array(proprio_state)

    match obs_type:
        case 'state':
            return torch.tensor(crop_obs_for_env(observation, env_name), device=device, dtype=torch.float32)
        case 'proprio':
            return torch.tensor(crop_obs_for_env(observation, env_name, proprio=True), device=device, dtype=torch.float32)
        case 'r3m':
            if not isinstance(frame, list):
                frame = [frame]
                batch_size = 1
            else:
                batch_size = len(frame)

            frame = np.array(frame)

            assert frame.shape[2] % 224 == 0
            num_viewpoints = frame.shape[2] // 224
            split_frames = np.stack(np.split(frame, num_viewpoints, axis=2), axis=1).reshape((batch_size * num_viewpoints, 224, 224, 3))
            image_features = model.r3m.frames_to_r3m(split_frames).reshape((batch_size, RESNET_SIZE * num_viewpoints))

            return torch.hstack((torch.tensor(proprio_state, device=device, dtype=torch.float32), image_features))
        case 'rgb':
            return torch.as_tensor(np.hstack([proprio_state, cv2.resize(frame, (84, 84)).flatten()]), device=device, dtype=torch.float32)

def get_action_from_obs_batched(config, model, envs, observations, frames, obs_history=None):
    if config.get('mixed'):
        obs = {}
        processed_obs_types = {}
        for dataset in ['retrieval', 'delta_state']:
            obs_type = config[dataset]['type']
            if obs_type not in processed_obs_types.keys():
                obs[dataset] = get_processed_obs(observations, frames, envs, model, config, obs_type)

                processed_obs_types[obs_type] = dataset
            else:
                obs[dataset] = (obs[processed_obs_types[obs_type]]).detach().clone()

        obs = torch.hstack((obs['retrieval'], obs['delta_state']))
    else:
        obs_type = config['type']
        
        obs = get_processed_obs(observations, frames, envs, model, config, obs_type)

    if hasattr(model, "get_action"):
        actions = model.get_action(obs, curr_rgb_obs=cv2.resize(frame, (224, 224), cv2.INTER_AREA).flatten()).squeeze()
    else:
        if obs_history is not None:
            if obs_history.shape[2] == 0:
                obs_history = torch.empty((obs_history.shape[0], 0, obs.shape[-1]), device=obs_history.device)
            obs_history = torch.cat((obs_history, obs.unsqueeze(1)), dim=1)
            obs_horizon = getattr(model, "obs_horizon", 1)
            if obs_horizon > 1 and not hasattr(model, 'retrieval_agent'):
                if obs_history.shape[1] < obs_horizon:
                    padding_needed = obs_horizon - obs_history.shape[1]
                    padding = obs_history[:, 0].unsqueeze(1).repeat(1, padding_needed, 1)
                    full_obs_history = torch.cat((padding, obs_history), dim=1)
                else:
                    full_obs_history = obs_history[:, -obs_horizon:]

                flat_obs_history = full_obs_history.reshape(full_obs_history.shape[0], -1)
                    
                actions = model(flat_obs_history)
            else:
                actions = model(obs_history)
        else:
            actions = model(obs)

    return actions.cpu().detach().numpy(), obs_history

def env_to_rgb_array(env, camera, crop_corners, width, height):
    crop_width = crop_corners[1][0] - crop_corners[0][0]
    render_width = width / crop_width

    crop_height = crop_corners[1][1] - crop_corners[0][1]
    render_height = height / crop_height

    render_size = max(render_width, render_height)

    frame = env.render(mode='rgb_array', height=round(render_size), width=round(render_size), camera_name=camera)
    assert frame is not None

    crop_corners[:, 0] *= render_size
    crop_corners[:, 1] *= render_size
    crop_corners = np.round(crop_corners).astype(np.uint16)
    cropped_frame = frame[crop_corners[0][1]:crop_corners[1][1], crop_corners[0][0]:crop_corners[1][0], :]
    return cv2.resize(cropped_frame, (height, width))

def eval_over(steps, config, env_instance):
    ENV_MAX_STEPS = {
        "push_t": 200,
        "hopper-expert-v2": 1000,
        "ant-expert-v2": 1000,
        "walker2d-expert-v2": 1000,
        "halfcheetah-expert-v2": 1000,
        "maze2d-umaze-v1": 500,
        "Stack_D0": 200,
        "Square_D0": 200,
        "StackThree_D0": 350,
        "ThreePieceAssembly_D0": 400,
        "Threading_D0": 300,
        "CloseDrawer": 300,
        "PickPlaceCounterToMicrowave": 600,
        "PickPlaceCounterToSink": 700,
        "CloseSingleDoor": 400,
        "TurnOnStove": 400,
    }

    env_name = config['name']
    max_steps = ENV_MAX_STEPS.get(env_name)
    
    if max_steps is None:
        return False
    
    if env_name == "maze2d-umaze-v1":
        at_target = np.linalg.norm(env_instance._get_obs()[0:2] - env_instance._target) <= 0.5
        return at_target or steps >= max_steps
    
    return steps >= max_steps

def crop_obs_for_env(obs, env, proprio=False):
    ROBOSUITE_TASKS = {
    "Square_D0", "Square_D1", "Stack_D0", "PickAndPlace_D0",
    "Threading_D0", "ThreePieceAssembly_D0", "StackThree_D0",
    }
    CASA_SINGLE = {"CloseDrawer", "CloseSingleDoor", "TurnOnStove", "TurnOffStove"}
    CASA_PNP = {"PickPlaceCounterToMicrowave", "PickPlaceCounterToSink"}

    ROBOSUITE_LOW_DIM_OBS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
    if env == "ant-expert-v2":
        return np.array(obs)[:, :27]
    
    if env == "coffee-pull-v2" or env == "coffee-push-v2":
        return np.concatenate((obs[:11], obs[18:29], obs[-3:]))
    
    if env == "button-press-topdown-v2":
        return np.concatenate((obs[:9], obs[18:27], obs[-2:]))
    
    if env == "drawer-close-v2":
        return np.concatenate((obs[:7], obs[18:25], obs[-3:]))
    
    if env in ROBOSUITE_TASKS | CASA_SINGLE | CASA_PNP:
        all_obs = []
        for o in obs:
            ret_obs = np.array([])
            for key in ROBOSUITE_LOW_DIM_OBS:
                if key == "object" and proprio:
                    continue
                if key == "object" and env in CASA_SINGLE:
                    ret_obs = np.hstack((ret_obs, o[key][:14]))
                elif key == "object" and env in CASA_PNP:
                    ret_obs = np.hstack((ret_obs, o[key][:42]))
                else:
                    ret_obs = np.hstack((ret_obs, o[key]))
            all_obs.append(ret_obs)
        return np.array(all_obs, dtype=np.float32)
    
    return np.array(obs)
