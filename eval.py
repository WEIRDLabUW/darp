import os

from rgb_arrays_to_mp4 import rgb_arrays_to_mp4
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
import numpy as np
import pickle
import torch
import torch.multiprocessing as mp
from eval_util import construct_env, eval_over, get_action_from_obs_batched, env_to_rgb_array, get_processed_obs
from util import set_seed
import torch.distributed as dist
import queue
import traceback
import psutil
from scipy.spatial.transform import Rotation as R
from logging_util import logger
import math
from pathlib import Path

persistent_processes = [None] * 8
class PersistentProcessPool:
    def __init__(self, config, start_trial, num_workers, rank, trials_per_worker, total_trials):
        self.command_queues = [mp.Queue() for _ in range(num_workers)]
        self.result_queue = mp.Queue()
        self.config = config
        self.workers = []
        self.start_trial = start_trial
        self.rank = rank
        self.lock = mp.Lock()

        for i in range(num_workers):
            trials = list(range((i * trials_per_worker) + start_trial, min((i + 1) * trials_per_worker + start_trial, total_trials + start_trial)))
            p = mp.Process(target=env_worker, 
                          args=(i, trials, self.config, self.command_queues[i], 
                               self.result_queue, rank, True, self.lock))
            p.start()
            self.workers.append(p)

        import signal
        import atexit
        atexit.register(self.shutdown)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self):
        self.shutdown()
        exit(0)

    def shutdown(self):
        for com_queue in self.command_queues:
            com_queue.put(None)
        for worker in self.workers:
            worker.join()

def single_trial_eval(config, agent, env, trial, reset, darp):
    env_name = config['name']
    is_robosuite = config.get('robosuite', False)
    cam_names = config.get("cams", [])
    crops = config.get('crops', {})
    obs_horizon = getattr(agent, "obs_horizon", 1)
    
    video_frames = []

    if not is_robosuite:
        env.seed(trial)

    if reset:
        observation = env.reset()
    else:
        if is_robosuite:
            observation = env.get_observation()
        else:
            observation = env.env._wrapped_env._get_obs()

    if env_name == "maze2d-umaze-v1":
        env.set_target()

    if config['name'] == "maze2d-umaze-v1":
        observation = np.hstack((env._target, observation))

    if (darp and agent.retrieval_agent.lookback > 1) or obs_horizon > 1:
        obs_history = torch.empty((1, 0, 0), device=agent.device)
    else:
        obs_history = None

    steps = 0
    episode_reward = 0
    done = False
    while not (steps > 0 and (done or eval_over(steps, config, env))):
        if True:
            height, width = 1024, 1024
        else:
            height, width = 224, 224

        if len(cam_names) > 0:
            full_frame = np.empty((height, 0, 3), dtype=np.uint8)
            for camera in cam_names:
                crop_corners = np.array(crops.get(camera, [[0, 0], [1.0, 1.0]]))
                frame = env_to_rgb_array(env, camera, crop_corners, width, height)

                full_frame = np.hstack((full_frame, frame))
        else:
            full_frame = np.empty((height, 0, 3), dtype=np.uint8)

        if config['name'] == "TurnOffStove" or config['name'] == "TurnOnStove":
            knob_id = env.env.sim.model.geom_name2id(f"{env.env.stove.name}_knob_{env.env.knob}_main")
            knob_pos = np.array(env.env.sim.data.geom_xpos[knob_id])
            observation['object'] = knob_pos
        elif config['name'] == "PickPlaceCounterToSink":
            sink_id = env.env.sim.model.geom_name2id(f"{env.env.sink.name}_bottom")
            sink_pos = np.array(env.env.sim.data.geom_xpos[sink_id])

            obj_id = env.env.sim.model.body_name2id(env.env.objects["obj"].root_body)
            obj_pos = np.array(env.env.sim.data.body_xpos[obj_id])
            obj_mat = R.from_matrix(np.array(env.env.sim.data.body_xmat[obj_id].reshape(3, 3)))
            obj_quat = obj_mat.as_quat()

            observation['object'] = np.hstack((sink_pos, obj_pos, obj_quat))

        with torch.no_grad():
            if not (darp and agent.retrieval_agent.lookback > 1):
                action, obs_history = get_action_from_obs_batched(config, agent, [env], [observation], [full_frame], obs_history=obs_history, numpy_action=False, is_first_ob=(steps == 0))
            else:
                action, _ = get_action_from_obs_batched(config, agent, [env], [observation], [full_frame], obs_history=None, numpy_action=False, is_first_ob=(steps == 0))

                observation = get_processed_obs(observation, full_frame, env, agent, config, config['type'])[0]
                if obs_history.shape[2] == 0:
                    obs_history = torch.empty((obs_history.shape[0], 0, observation.shape[-1]), device=obs_history.device)
                obs_history = torch.cat((obs_history, observation.unsqueeze(0).unsqueeze(0)), dim=1)


        action = action[0]

        video_frames.append(full_frame)

        observation, reward, done, info = env.step(action)[:4]
        if hasattr(env, "env") and hasattr(env.env, "_check_success"):
            reward = 1.0 if env.env._check_success() else 0.0
            done = done or reward == 1.0

        if config['name'] == "maze2d-umaze-v1":
            observation = np.hstack((env._target, observation))

        if env_name == "push_t":
            episode_reward = max(episode_reward, reward)
        else:
            episode_reward += reward
            if is_robosuite and ((not config.get('reward_shaping', False) and episode_reward > 0) or (config.get('reward_shaping', False) and done)):
                break

        steps += 1

    if len(video_frames) > 0:
        video_frames = np.array(video_frames)
        Path(f"vids/{env_name}").mkdir(parents=True, exist_ok=True)
        rgb_arrays_to_mp4(video_frames, f"vids/{env_name}/{trial}_{'darp' if darp else 'bc'}.mp4")

    success = 1 if 'success' in info else 0

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    return episode_reward, success

def prepare_env(config, trial=None, gpu_id=0, lock=None):
    import robosuite.renderers.context.egl_context as egl_context
    egl_context.EGL_DISPLAY = None

    import robomimic

    set_seed(42)
    env = construct_env(config, gpu_id=gpu_id, seed=trial, lock=lock)
    env_name = config['name']
    is_robosuite = config.get('robosuite', False)

    if not is_robosuite:
        env.seed(trial)
    elif is_robosuite and not robomimic.__version__ == "0.3.0":
        for _ in range(trial):
            env.reset()

    if env_name == "maze2d-umaze-v1":
        env.set_target()

    return env

def env_worker(worker_id, trials, config, command_queue, result_queue, local_rank, reset, lock):
    import os
    p = psutil.Process()
    available_cpus = list(os.sched_getaffinity(0))
    p.cpu_affinity([available_cpus[worker_id % len(available_cpus)]])

    os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)
    is_robosuite = config.get('robosuite', False)
    env = None

    initial_states = [None] * len(trials)

    try:
        while True:
            command = command_queue.get()
            if command is None:
                break
                
            cmd_type, data = command
            
            if cmd_type == 'step':
                assert env is not None
                action, step_num = data
                observation, reward, done, info = env.step(action)[:4]

                if config['name'] == "maze2d-umaze-v1":
                    observation = np.hstack((env._target, observation))

                if is_robosuite and not config.get('reward_shaping', False) and reward > 0:
                    done = True

                done = done or eval_over(step_num, config, env)

                if hasattr(env, "env") and hasattr(env.env, "_check_success"):
                    reward = 1.0 if env.env._check_success() else 0.0
                    done = done or reward == 1.0

                if done and not is_robosuite:
                    del env
                    env = None

                if config['name'] == "TurnOffStove" or config['name'] == "TurnOnStove":
                    knob_id = env.env.sim.model.geom_name2id(f"{env.env.stove.name}_knob_{env.env.knob}_main")
                    knob_pos = np.array(env.env.sim.data.geom_xpos[knob_id])
                    observation['object'] = knob_pos
                elif config['name'] == "PickPlaceCounterToSink":
                    sink_id = env.env.sim.model.geom_name2id(f"{env.env.sink.name}_bottom")
                    sink_pos = np.array(env.env.sim.data.geom_xpos[sink_id])

                    obj_id = env.env.sim.model.body_name2id(env.env.objects["obj"].root_body)
                    obj_pos = np.array(env.env.sim.data.body_xpos[obj_id])
                    obj_mat = R.from_matrix(np.array(env.env.sim.data.body_xmat[obj_id].reshape(3, 3)))
                    obj_quat = obj_mat.as_quat()

                    observation['object'] = np.hstack((sink_pos, obj_pos, obj_quat))

                result_queue.put((worker_id, 'step_result', {
                    'observation': observation,
                    'reward': reward, 
                    'done': done,
                    'info': info
                }))
            elif cmd_type == 'init_trial':
                trial_idx = data

                if trial_idx >= len(trials):
                    logger.debug(f"{local_rank}:{worker_id} Sending worker unneeded")
                    result_queue.put((worker_id, 'worker_unneeded', None))
                    continue
                elif is_robosuite and env is not None and initial_states[trial_idx] is not None:
                    logger.debug(f"{local_rank}:{worker_id} loading state for trial {trials[trial_idx]}")
                    env.reset_to(initial_states[trial_idx])
                elif config['name'] == "push_t" and env is not None:
                    env._set_state(initial_state)
                else:
                    if env is not None:
                        del env

                    logger.debug(f"{local_rank}:{worker_id} creating env for trial {trials[trial_idx]} for the first time")
                    env = prepare_env(config, trial=trials[trial_idx], gpu_id=local_rank, lock=lock)

                    if is_robosuite:
                        env.reset()
                        initial_states[trial_idx] = env.get_state()
                        initial_states[trial_idx]['model'] = env.env.sim.model.get_xml()
                        env.reset_to(initial_states[trial_idx])
                        reset = False
                    elif config['name'] == "push_t":
                        env.reset()
                        initial_states[trial_idx] = env._get_obs()
                        env._set_state(initial_states[trial_idx])
                        reset = False

                if reset:
                    observation = env.reset()

                    if config['name'] == "push_t":
                        observation = observation[0]
                else:
                    observation = env.get_observation()

                if config['name'] == "maze2d-umaze-v1":
                    observation = np.hstack((env._target, observation))

                if config['name'] == "TurnOffStove" or config['name'] == "TurnOnStove":
                    knob_id = env.env.sim.model.geom_name2id(f"{env.env.stove.name}_knob_{env.env.knob}_main")
                    knob_pos = np.array(env.env.sim.data.geom_xpos[knob_id])
                    observation['object'] = knob_pos
                elif config['name'] == "PickPlaceCounterToSink":
                    sink_id = env.env.sim.model.geom_name2id(f"{env.env.sink.name}_bottom")
                    sink_pos = np.array(env.env.sim.data.geom_xpos[sink_id])

                    obj_id = env.env.sim.model.body_name2id(env.env.objects["obj"].root_body)
                    obj_pos = np.array(env.env.sim.data.body_xpos[obj_id])
                    obj_mat = R.from_matrix(np.array(env.env.sim.data.body_xmat[obj_id].reshape(3, 3)))
                    obj_quat = obj_mat.as_quat()

                    observation['object'] = np.hstack((sink_pos, obj_pos, obj_quat))

                result_queue.put((worker_id, 'env_created', observation))
                
            elif cmd_type == 'get_frame':
                camera, crop_corners, width, height = data
                frame = env_to_rgb_array(env, camera, crop_corners, width, height)
                result_queue.put((worker_id, 'frame', frame))
                
    except Exception as e:
        logger.debug(f"{local_rank}:{worker_id} ERROR: {traceback.format_exc()}")
        result_queue.put((worker_id, 'error', traceback.format_exc()))

def batched_eval(config, agent, trials=10, results=None, reset=False, darp=True, trials_per_worker=1):
    global persistent_processes
    mp.set_start_method('spawn', force=True)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    set_seed(42)

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    trials_per_proc = trials // world_size
    remainder = trials % world_size

    my_num_trials = trials_per_proc + (1 if rank < remainder else 0)
    start_trial = rank * trials_per_proc + min(rank, remainder)
    end_trial = start_trial + my_num_trials

    my_num_workers = math.ceil(my_num_trials / trials_per_worker)

    logger.info(f"GPU {local_rank} taking trials {start_trial + 1} to {end_trial}")

    env_name = config['name']
    cam_names = config.get("cams", [])
    crops = config.get('crops', {})
    height, width = 224, 224

    command_queues = [mp.Queue() for _ in range(my_num_trials)]
    result_queue = mp.Queue()

    obs_horizon = getattr(agent, "obs_horizon", 1)
        
    if persistent_processes[local_rank] == None:
        logger.info("Creating PersistentProcessPool")
        persistent_processes[local_rank] = PersistentProcessPool(config, start_trial, my_num_workers, local_rank, trials_per_worker, my_num_trials)

    command_queues = persistent_processes[local_rank].command_queues
    result_queue = persistent_processes[local_rank].result_queue

    # Inexplicably, without this line I get CUBLAS_STATUS_ALLOC_FAILED downstream
    test = torch.randn(1, 1, device=local_rank)
    torch.matmul(test, test.T)

    all_episode_rewards = []
    for trial_idx in range(trials_per_worker):

        for i in range(my_num_workers):
            command_queues[i].put(('init_trial', trial_idx))

        observations = [None] * my_num_workers
        envs_created = 0
        
        dones = np.zeros(my_num_workers).astype(bool)
        episode_rewards = np.zeros(my_num_workers)
        while envs_created < my_num_workers:
            try:
                worker_id, msg_type, data = result_queue.get()
            except queue.Empty:
                logger.debug("QUEUE EMPTY")
                break

            if msg_type == 'env_created':
                observations[worker_id] = data
                envs_created += 1
                logger.debug(f"{rank} Environment {worker_id + 1 + start_trial}/{trials} created, {(envs_created/my_num_workers) * 100:.0f}% of my envs created")
            elif msg_type == 'worker_unneeded':
                dones[worker_id] = True
                episode_rewards[worker_id] = -1
                envs_created += 1
                logger.debug(f"{rank} Environment {worker_id + 1 + start_trial}/{trials} unneeded, {(envs_created/my_num_workers) * 100:.0f}% of my envs created")
            elif msg_type == 'error':
                logger.error(f"Error in worker {worker_id}: {data}. Will retry.")

        # [B, H, O]
        if (darp and agent.retrieval_agent.lookback > 1) or obs_horizon > 1:
            obs_history = torch.empty((my_num_workers, 0, 0), device=local_rank)

        try:
            steps = 0

            while not (steps > 0 and np.all(dones)):
                frames = None
                if len(cam_names) > 0:
                    for i, com_queue in enumerate(command_queues):
                        if not dones[i]:
                            for camera in cam_names:
                                crop_corners = np.array(crops.get(camera, [[0, 0], [1.0, 1.0]]))
                                com_queue.put(('get_frame', (camera, crop_corners, width, height)))
                    frames = [[] for _ in range(my_num_workers)]
                    expected_frames = sum(len(cam_names) for i in range(my_num_workers) if not dones[i])
                    for _ in range(expected_frames):
                        worker_id, msg_type, frame = result_queue.get()
                        frames[worker_id].append(frame)
                    
                    for i in range(my_num_workers):
                        if frames[i]:
                            frames[i] = np.hstack(frames[i])

                active_envs = [i for i in range(my_num_workers) if not dones[i]]
                if not active_envs:
                    break
                    
                active_observations = [observations[i] for i in active_envs]
                active_frames = [frames[i] for i in active_envs] if frames else None

                with torch.no_grad():
                    if (darp and agent.retrieval_agent.lookback > 1) or obs_horizon > 1:
                        actions, new_obs_history = get_action_from_obs_batched(config, agent, active_envs, active_observations, active_frames, obs_history=obs_history[active_envs])
                        if steps == 0:
                            obs_history = torch.zeros((my_num_workers, 1, new_obs_history.shape[-1]), device=agent.device)
                            obs_history[active_envs] = new_obs_history
                        else:
                            full_new_obs_history = torch.zeros((my_num_workers, 1, new_obs_history.shape[-1]), device=agent.device)
                            full_new_obs_history[active_envs] = new_obs_history[:, -1].unsqueeze(1)
                            obs_history = torch.cat((obs_history, full_new_obs_history), dim=1)
                    else:
                        actions, _ = get_action_from_obs_batched(config, agent, active_envs, active_observations, active_frames)

                for idx, action in zip(active_envs, actions):
                    command_queues[idx].put(('step', (action, steps)))

                for i in range(len(active_envs)):
                    worker_id, msg_type, result = result_queue.get()
                    if not isinstance(result, dict):
                        logger.error(f"Step failed on worker {worker_id} with message {result}")
                    observations[worker_id] = result['observation']
                    if env_name == "push_t":
                        episode_rewards[worker_id] = max(episode_rewards[worker_id], result['reward'])
                    else:
                        episode_rewards[worker_id] += result['reward']
                    dones[worker_id] = result['done']

                steps += 1


        finally:
            pass
            # for queue in command_queues:
            #     queue.put(None)  # Shutdown signal
            #
            # for worker in workers:
            #     worker.join()

        episode_rewards = episode_rewards[episode_rewards >= 0]
        if world_size > 1:
            # Create tensors to gather results
            all_rewards = [None for _ in range(world_size)]

            # Gather rewards and successes
            dist.all_gather_object(all_rewards, episode_rewards)

            # Flatten rewards list and sum successes
            all_episode_rewards.extend([r for proc_rewards in all_rewards for r in proc_rewards])
        else:
            all_episode_rewards.extend(episode_rewards)

    # Save results (only on rank 0)
    if rank == 0:
        logger.info(all_episode_rewards)
        if results is not None:
            os.makedirs('results', exist_ok=True)
            with open(f"results/{results}.pkl", 'wb') as f:
                pickle.dump(all_episode_rewards, f)

        logger.info(f"mean {round(np.mean(all_episode_rewards), 2)}, std {round(np.std(all_episode_rewards), 2)}")

    # Wait for all processes
    if world_size > 1:
        dist.barrier()

    return np.mean(all_episode_rewards)

def parallel_eval(config, nn_agent, trials=10, results=None, darp=False):
    # Initialize the distributed environment
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    # Set up the process group
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        nn_agent = nn_agent.to(local_rank)

    # Construct environment on each process
    set_seed(42)
    
    # Divide trials among processes
    trials_per_proc = trials // world_size
    remainder = trials % world_size

    # Distribute remaining trials evenly
    my_num_trials = trials_per_proc + (1 if rank < remainder else 0)
    start_trial = rank * trials_per_proc + min(rank, remainder)

    env = construct_env(config, gpu_id=local_rank, seed=start_trial, render=True)

    # No seeding, so this is the only way to make sure we don't do repeat seeds 
    if config.get('robosuite', False):
        for _ in range(start_trial):
            env.reset()

    end_trial = start_trial + my_num_trials

    logger.info(f"GPU {local_rank} taking trials {start_trial + 1} to {end_trial}")

    # Run assigned trials
    episode_rewards = []
    successes = 0

    for trial in range(start_trial, end_trial):
        episode_reward, success = single_trial_eval(config, nn_agent, env, trial, reset=True, darp=darp)

        episode_rewards.append(episode_reward)
        successes += success

    # Gather results from all processes
    if world_size > 1:
        # Create tensors to gather results
        all_rewards = [None for _ in range(world_size)]
        all_successes = [None for _ in range(world_size)]

        # Gather rewards and successes
        dist.all_gather_object(all_rewards, episode_rewards)
        dist.all_gather_object(all_successes, successes)

        # Flatten rewards list and sum successes
        episode_rewards = [r for proc_rewards in all_rewards for r in proc_rewards]
        successes = sum(all_successes)

    # Save results (only on rank 0)
    if rank == 0:
        if results is not None:
            os.makedirs('results', exist_ok=True)
            with open(f"results/{results}.pkl", 'wb') as f:
                pickle.dump(episode_rewards, f)

        logger.debug(episode_rewards)
        logger.info(f"mean {round(np.mean(episode_rewards), 2)}, std {round(np.std(episode_rewards), 2)}")

    # Wait for all processes
    if world_size > 1:
        dist.barrier()

    return np.mean(episode_rewards)
