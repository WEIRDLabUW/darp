import h5py
import argparse
import numpy as np
import pickle
import os
import json
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser()
parser.add_argument(
    "file",
    type=str,
    help="Path to your hdf5 file"
)
parser.add_argument('--goal', action=argparse.BooleanOptionalAction)

args = parser.parse_args()

f = h5py.File(args.file, "r")
demos = f['data']
expert_data = []
traj_lengths = []
print(demos['demo_0']['obs'].keys())
for demo in demos:
    #traj_lengths.append(len(f['data'][demo]['obs']['object']))
#for demo in np.array(demos)[np.argsort(traj_lengths)]:
    # Construct observation
    #<KeysViewHDF5 ['agentview_image', 'object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_eef_vel_ang', 'robot0_eef_vel_lin', 'robot0_eye_in_hand_image', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel']>

    default_low_dim_obs = [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "object",
    ]

    demo_obs = f['data'][demo]['obs']
    obs = np.empty((len(f['data'][demo]['obs'][default_low_dim_obs[0]]), 0))

    for key in default_low_dim_obs:
        if key == 'object' and not args.goal:
            continue

        #print(f"{key}: {demo_obs[key]}")
        #print(f"{key}: {len(demo_obs[key][0])}")
        #if key == 'object':
            #print(f"{len(demo_obs[key][0]) / 14}")
            #print(f"{demo_obs[key][0][:14]}")
            #print(f"{demo_obs[key][0]}")
        
        if key == 'object':
            #obs = np.hstack((obs, demo_obs[key]))

            # For casa
            #obs = np.hstack((obs, demo_obs[key][:, :14]))
            #obs = np.hstack((obs, demo_obs[key][:, :14 * 3]))
            #breakpoint()
            # Stove
            #knob_pos = f['data'][demo]['datagen_info']['object_poses']['knob'][..., :3, 3]
            #obs = np.hstack((obs, knob_pos))

            # ToSink
            sink_pos = f['data'][demo]['datagen_info']['object_poses']['sink'][..., :3, 3]
            obj_pos = f['data'][demo]['datagen_info']['object_poses']['obj'][..., :3, 3]
            obj_mat = R.from_matrix(f['data'][demo]['datagen_info']['object_poses']['obj'][..., :3, :3])
            obj_quat = obj_mat.as_quat()

            obs = np.hstack((obs, sink_pos, obj_pos, obj_quat))
        else:
            obs = np.hstack((obs, demo_obs[key]))

    actions = f['data'][demo]['actions']
    states = f['data'][demo]['states']


    #print(len(json.loads(demos[demo].attrs['ep_meta'])['object_cfgs']))
    #print(json.loads(demos[demo].attrs['ep_meta'])['object_cfgs'][0])
    #print(json.loads(demos[demo].attrs['ep_meta'])['object_cfgs'])
    expert_data.append({"observations": np.array(obs[:], dtype=np.float32), "actions": np.array(actions[:], dtype=np.float32), "states": np.array(states[:], dtype=np.float32), "model_file": demos[demo].attrs['model_file']})

if args.goal:
    file_name = f"{os.path.dirname(args.file)}/{len(expert_data)}.pkl"
else:
    file_name = f"{os.path.dirname(args.file)}/{len(expert_data)}_proprio.pkl"

print(f"Dumping to {file_name}...")
pickle.dump(expert_data, open(file_name, 'wb'))

