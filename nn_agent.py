import math
import torch

from util import load_and_scale_data, set_seed
from logging_util import logger

def compute_distance_with_rot(curr_ob: torch.Tensor, flattened_obs_matrix: torch.Tensor, rot_weights: torch.Tensor):
    delta = torch.abs(curr_ob - flattened_obs_matrix)
    wrapped_delta = torch.min(delta, 2 * torch.pi - delta) / (2 * torch.pi)
    neighbor_vec_distances = wrapped_delta * rot_weights
    
    squared_dists = torch.sum(neighbor_vec_distances ** 2, dim=1)
    
    neighbor_distances = torch.sqrt(squared_dists)
    
    return neighbor_distances

def compute_accum_distance(nearest_neighbors, max_lookbacks, obs_history, sequence_lengths, flattened_obs_matrix, decay_factors, distance_metric):
    b, m = nearest_neighbors.shape
    device = nearest_neighbors.device
    max_seq_len = max_lookbacks.max().item()

    seq_indices = torch.arange(max_seq_len, device=device).flip(0).view(1, 1, -1)
    gather_indices = torch.maximum((nearest_neighbors).unsqueeze(2) - seq_indices, torch.tensor(0))
    matrix_slices = flattened_obs_matrix[gather_indices]

    obs_indices = obs_history.shape[1] - 1 - seq_indices
    obs_expanded = obs_history[torch.arange(b, device=device).view(b, 1, 1).expand(-1, m, max_seq_len), obs_indices]

    valid_mask = (seq_indices < max_lookbacks.unsqueeze(2)) & (seq_indices < sequence_lengths.view(b, 1, 1))
    feature_mask = valid_mask.unsqueeze(3)
    
    if distance_metric == "cosine":
        # Calculate cosine similarity along the feature dimension (dim=3)
        # Result shape: (B, M, T)
        sim = torch.nn.functional.cosine_similarity(obs_expanded, matrix_slices, dim=3, eps=1e-6)
        
        # Convert similarity [-1, 1] to distance [0, 2]
        distances = 1.0 - sim
    else:
        diff = (obs_expanded - matrix_slices) * feature_mask
        distances = torch.sqrt(torch.sum(diff * diff, dim=3))
    distances[torch.isnan(distances)] = 0
    weighted_distances = distances * decay_factors[-max_seq_len:].view(1, 1, -1) * valid_mask

    return torch.sum(weighted_distances, dim=2)

class NN_METHOD:
    NN, NS, KNN, KNN_AND_DELTA = range(4)

    @classmethod
    def from_string(cls, name):
        match name:
            case 'nn':
                return NN_METHOD.NN
            case 'ns':
                return NN_METHOD.NS
            case 'knn':
                return NN_METHOD.KNN
            case 'knn_and_delta':
                return NN_METHOD.KNN_AND_DELTA
            case _:
                logger.warning(f"No such method {name}! Defaulting to NN")
                return NN_METHOD.NN

class NNAgent:
    def __init__(self, env_cfg, policy_cfg):
        set_seed(env_cfg.get("seed", 42))
        self.env_cfg = env_cfg
        self.policy_cfg = policy_cfg

        self.device = env_cfg['device']
        self.to_device(self.device)
        self.method = NN_METHOD.from_string(policy_cfg.get('method'))

        # If this is already defined, a subclass has intentionally set it
        if not hasattr(self, 'datasets'):
            self.datasets = {}
            # We may use different datasets for retrieval, neighbor state, and state delta
            if env_cfg.get('mixed', False):
                # Lookup dict for duplicate datasets
                paths = {}
                for dataset in ['retrieval', 'state', 'delta_state']:
                    path = env_cfg[dataset]['demo_pkl']

                    # Check for duplicates
                    if path in paths.keys():
                        self.datasets[dataset] = self.datasets[paths[path]]
                    else:
                        paths[path] = dataset

                        self.datasets[dataset] = load_and_scale_data(
                            path,
                            env_cfg[dataset].get('rot_indices', []),
                            env_cfg[dataset].get('weights', []),
                            use_torch=True,
                            scale=False
                        )
            else:
                expert_data_path = env_cfg['demo_pkl']
                one_dataset = load_and_scale_data(
                    expert_data_path,
                    env_cfg.get('rot_indices', []),
                    env_cfg.get('weights', []),
                    ob_type=env_cfg.get('type', 'state'),
                    use_torch=True,
                    scale=False
                )

                for dataset in ['retrieval', 'state', 'delta_state']:
                    self.datasets[dataset] = one_dataset

        self.rot_indices = self.datasets['retrieval'].rot_indices
        self.non_rot_indices = self.datasets['retrieval'].non_rot_indices

        self.candidates = policy_cfg.get('k', 100)
        self.lookback = policy_cfg.get('lookback', 10)
        self.decay = policy_cfg.get('decay_rate', 1)
        self.window = policy_cfg.get('dtw_window', 0)
        self.final_neighbors_ratio = policy_cfg.get('final_neighbors_ratio', 1)
        self.obs_horizon = policy_cfg.get('obs_horizon', 1)
        self.distance_metric = policy_cfg.get('distance_metric', "euclidean")

        self.env_cfg = env_cfg

        # Precompute constants
        self.obs_history = torch.tensor([], dtype=torch.float32)

        self.i_array = torch.arange(self.lookback, 0, -1, dtype=torch.float32)
        self.decay_factors = torch.pow(self.i_array, self.decay)

class NNAgentEuclidean(NNAgent):
    def get_neighbors(self, current_ob):
        # Batched input [b, k, n]
        is_batched = current_ob.dim() == 3
        batch_size = current_ob.shape[0] if is_batched else 1
        if not is_batched:
            current_ob = current_ob.unsqueeze(0)
        self.obs_history = current_ob
        
        first_features = current_ob[:, :, 0]  # [b, k]
        valid_mask = ~torch.isnan(first_features)  # [b, k] - True where observations are valid
        sequence_lengths = torch.sum(valid_mask, dim=1)  # [b] - count of valid observations per batch

        current_ob = current_ob[:, -1]

        # If we have elements in our observation space that wraparound (rotations), we can't just do direct Euclidean distance
        if not hasattr(self, '_weighted_ob_buffer') or self._weighted_ob_buffer.shape[0] != batch_size:
            self._weighted_ob_buffer = torch.empty(batch_size, current_ob.shape[-1], device=current_ob.device, dtype=current_ob.dtype)

        # Copy and apply weights in one operation
        torch.mul(
            current_ob,
            self.datasets['retrieval'].weights.to(self.device),
            out=self._weighted_ob_buffer
        )

        if self.distance_metric == "cosine":
            all_distances = 1 - torch.nn.functional.cosine_similarity(self.datasets['retrieval'].processed_obs_matrix[:, self.non_rot_indices].unsqueeze(0), self._weighted_ob_buffer[:, self.non_rot_indices].unsqueeze(1), dim=2)
        else:
            all_distances = torch.sqrt(torch.sum(torch.pow(torch.subtract(self.datasets['retrieval'].processed_obs_matrix[:, self.non_rot_indices].unsqueeze(0), self._weighted_ob_buffer[:, self.non_rot_indices].unsqueeze(1)), 2), dim=2))

        if len(self.rot_indices) > 0:
            all_distances += compute_distance_with_rot(self._weighted_ob_buffer[self.rot_indices], self.datasets['retrieval'].processed_obs_matrix[:, self.rot_indices], self.datasets['retrieval'].weights[self.datasets['retrieval'].rot_indices])

        # When training, don't include the state itself
        if self.method == NN_METHOD.KNN or self.method == NN_METHOD.KNN_AND_DELTA:
            zero_mask = (all_distances == 0.0)
            all_distances[zero_mask] = torch.inf

        _, nearest_neighbor_indices = torch.topk(all_distances, k=self.candidates, largest=False, dim=1)
        nearest_neighbors = nearest_neighbor_indices.to(torch.int32).to(self.device)

        # Find corresponding trajectories for each neighbor
        self.datasets['retrieval'].traj_starts = self.datasets['retrieval'].traj_starts.to(self.device)

        flat_neighbors = nearest_neighbors.reshape(-1)
        flat_traj_nums = torch.searchsorted(self.datasets['retrieval'].traj_starts, flat_neighbors, right=True) - 1
        traj_nums = flat_traj_nums.reshape(batch_size, self.candidates)

        flat_obs_nums = flat_neighbors - self.datasets['retrieval'].traj_starts[flat_traj_nums]
        obs_nums = flat_obs_nums.reshape(batch_size, self.candidates)
        if self.method == NN_METHOD.NN:
            # If we're doing direct nearest neighbor, just return that action
            nearest_neighbor_idx = torch.argmin(torch.gather(all_distances, 1, nearest_neighbors.long()), dim=1)
            batch_indices = torch.arange(batch_size, device=self.device)
            actions = self.datasets['retrieval'].act_matrix[traj_nums[batch_indices, nearest_neighbor_idx], obs_nums[batch_indices, nearest_neighbor_idx]]
            
            return actions.cpu().numpy() if is_batched else actions[0].cpu().numpy()

        if self.lookback == 1 or self.obs_history.shape[-2] == 1:
            # No lookback needed
            accum_distances = torch.gather(all_distances, 1, nearest_neighbors.long())
        else:
            # How far can we look back for each neighbor trajectory?
            # This is upper bound by min(lookback hyperparameter, length of obs history, neighbor distance into its traj)
            max_lookbacks = torch.minimum(
                torch.tensor(self.lookback, dtype=torch.int64, device=self.device),
                torch.minimum(
                    obs_nums + 1,
                    sequence_lengths.unsqueeze(1).expand(-1, self.candidates)
                )
            )
            
            accum_distances = compute_accum_distance(nearest_neighbors, max_lookbacks.to(self.device), self.obs_history.to(self.device), sequence_lengths, self.datasets['retrieval'].flattened_obs_matrix.to(self.device), self.decay_factors.to(self.device), self.distance_metric)

        if self.method == NN_METHOD.NS:
            # If we're doing direct nearest sequence, return that action
            nearest_sequence_idx = torch.argmin(accum_distances, dim=1)
            batch_indices = torch.arange(batch_size, device=self.device)
            actions = self.datasets['retrieval'].act_matrix[traj_nums[batch_indices, nearest_sequence_idx], obs_nums[batch_indices, nearest_sequence_idx]]

            return actions if is_batched else actions[0]
        # Do a final pass and pick only the top (self.final_neighbors_ratio * 100)% of neighbors based on this new accumulated distance
        final_neighbor_num = math.floor(accum_distances.shape[1] * self.final_neighbors_ratio)
        _, final_neighbor_indices = torch.topk(accum_distances, k=final_neighbor_num, largest=False, dim=1)

        batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1)
        final_neighbors = nearest_neighbors[batch_indices, final_neighbor_indices].to('cpu')

        if self.method == NN_METHOD.KNN:
            return final_neighbors if is_batched else final_neighbors[0]
        elif self.method == NN_METHOD.KNN_AND_DELTA:
            final_distances = torch.gather(accum_distances, 1, final_neighbor_indices)
            return final_neighbors if is_batched else final_neighbors[0], final_distances if is_batched else final_distances[0]

    def to_device(self, device):
        self.device = device
        if hasattr(self, "obs_history"):
            self.obs_history = self.obs_history.to(device)
        if hasattr(self, "_weighted_ob_buffer"):
            self._weighted_ob_buffer = self._weighted_ob_buffer.to(device)

        for dataset_type in self.datasets:
            dataset = self.datasets[dataset_type]

            dataset.obs_scaler.to_device(device)
            dataset.act_scaler.to_device(device)
            if isinstance(dataset.rot_indices, torch.Tensor):
                dataset.rot_indices = dataset.rot_indices.to(device)
            if isinstance(dataset.weights, torch.Tensor):
                dataset.weights = dataset.weights.to(device)
            if isinstance(dataset.traj_starts, torch.Tensor):
                dataset.traj_starts = dataset.traj_starts.to(device)
            if isinstance(dataset.flattened_obs_matrix, torch.Tensor):
                dataset.flattened_obs_matrix = dataset.flattened_obs_matrix.to(device)
            if isinstance(dataset.flattened_act_matrix, torch.Tensor):
                dataset.flattened_act_matrix = dataset.flattened_act_matrix.to(device)
            if isinstance(dataset.processed_obs_matrix, torch.Tensor):
                dataset.processed_obs_matrix = dataset.processed_obs_matrix.to(device)

# Standard Euclidean distance, but normalize each dimension of the observation space
class NNAgentEuclideanStandardized(NNAgentEuclidean):
    def __init__(self, env_cfg, policy_cfg):
        self.datasets = {}
        # We may use different datasets for retrieval, neighbor state, and state delta
        if env_cfg.get('mixed', False):
            # Lookup dict for duplicate datasets
            paths = {}
            for dataset in ['retrieval', 'state', 'delta_state']:
                path = env_cfg[dataset]['demo_pkl']

                # Check for duplicates
                if path in paths.keys():
                    self.datasets[dataset] = self.datasets[paths[path]]
                else:
                    paths[path] = dataset

                    self.datasets[dataset] = load_and_scale_data(
                        path,
                        env_cfg[dataset].get('rot_indices', []),
                        env_cfg[dataset].get('weights', []),
                        ob_type=env_cfg[dataset].get('type', 'state'),
                        device=env_cfg['device']
                    )
        else:
            expert_data_path = env_cfg['demo_pkl']
            for dataset in ['retrieval', 'state', 'delta_state']:
                one_dataset = load_and_scale_data(
                    expert_data_path,
                    env_cfg.get('rot_indices', []),
                    env_cfg.get('weights', []),
                    ob_type=env_cfg.get('type', 'state'),
                    device=env_cfg['device']
                )
                self.datasets[dataset] = one_dataset

        super().__init__(env_cfg, policy_cfg)

    def get_neighbors(self, current_ob, normalize=True):
        current_ob = torch.clone(current_ob) if torch.is_tensor(current_ob) else torch.from_numpy(current_ob, dtype=torch.float32, device=self.device)
        is_batched = current_ob.dim() == 3

        if normalize:
            dataset = self.datasets['retrieval']
            if is_batched:
                current_ob[:, :, dataset.non_rot_indices] = dataset.obs_scaler.transform(current_ob[:, :, dataset.non_rot_indices])
            else:
                current_ob[:, dataset.non_rot_indices] = dataset.obs_scaler.transform(current_ob[:, dataset.non_rot_indices])

        return super().get_neighbors(current_ob)

