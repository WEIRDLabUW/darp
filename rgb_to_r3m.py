import pickle
from argparse import ArgumentParser
from tensorflow_datasets.core.features import image_feature
import torch
import numpy as np
import torch.nn as nn


from constants import RESNET_SIZE
from train_bc import BCExpertDataset, ChunkingWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
from models.r3m import R3M

parser = ArgumentParser()
parser.add_argument("proprio_path")
parser.add_argument("rgb_path")
parser.add_argument("output_path")
parser.add_argument("--prop", action="store_true", default=False)

NUM_VIEWPOINTS = 4

args, _ = parser.parse_known_args()

data = pickle.load(open(args.rgb_path, 'rb'))

print("Getting chunked data...")
dataset = BCExpertDataset(args.rgb_path)

all_images = np.zeros((len(dataset) * NUM_VIEWPOINTS, 224 * 224 * 3), dtype=np.uint8)
vid_frames = []
vid_actions = []

for i in range(len(dataset)):
    item = dataset[i][0]
    for j in range(NUM_VIEWPOINTS):
        start = 224 * 224 * 3 * j
        end = 224 * 224 * 3 * (j + 1)
        all_images[(i * NUM_VIEWPOINTS) + j] = item[start:end]

print("Creating model...")
dummy_module = nn.Module()
dummy_module.output_len = 0
featurizer = R3M(dummy_module, **{'device': 'cuda', 'rgb_height': 224, 'rgb_width': 224, 'pretrained': True, 'grayscale': False})

print("Featurizing...")
img_features = featurizer.frames_to_r3m(all_images)
img_features = img_features.reshape((len(dataset), NUM_VIEWPOINTS * RESNET_SIZE))

i = 0
proprio_data = pickle.load(open(args.proprio_path, 'rb'))
for traj_num, traj in enumerate(data):
    print(f"Processing {traj_num}...")
    if args.prop:
        prop_observations = proprio_data[traj_num]['observations']
        traj["observations"] = torch.hstack((torch.tensor(prop_observations, device=device, dtype=torch.float32), img_features[i:i + len(traj["observations"])])).detach().cpu()
    else:
        traj["observations"] = img_features[i:i + len(traj["observations"])].detach().cpu()
    i += len(traj["observations"])

    if "states" in traj:
        traj["states"] = proprio_data[traj_num]['states']
    if "model_file" in traj:
        traj['model_file'] = proprio_data[traj_num]['model_file']

print(f"Success! Dumping to {args.output_path}")
pickle.dump(data, open(f"{args.output_path}", 'wb'))
