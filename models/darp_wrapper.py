from models.model_wrapper import ModelWrapper
from models.retrieval_wrapper import RetrievalAgent
import torch.nn as nn
import torch
from logging_util import logger
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64

class DARPWrapper(ModelWrapper):
    def __init__(self, wrapped: nn.Module, env_cfg, policy_cfg, **kwargs):
        super(DARPWrapper, self).__init__()

        self.wrapped = wrapped
        self.retrieval_agent = RetrievalAgent(env_cfg, policy_cfg)

        self.mixed = env_cfg.get("mixed", False) and env_cfg['retrieval']['demo_pkl'] != env_cfg['delta_state']['demo_pkl']

        self.input_splits = []
        self.input_splits.append(len(self.retrieval_agent.agent.datasets['retrieval'].obs_matrix[0][0]))
        self.input_splits.append(len(self.retrieval_agent.agent.datasets['delta_state'].obs_matrix[0][0]))
        self.input_splits = torch.cumsum(torch.tensor(self.input_splits), dim=0)
        logger.info(f"DARP input splits: {self.input_splits}")

        self.s_dataset = self.retrieval_agent.agent.datasets['state'].flattened_obs_matrix
        self.a_dataset = self.retrieval_agent.agent.datasets['state'].flattened_act_matrix
        self.state_size = len(self.retrieval_agent.agent.datasets['state'].flattened_obs_matrix[0])
        self.action_size = len(self.retrieval_agent.agent.datasets['state'].flattened_act_matrix[0])
        self.delta_s_dataset = self.retrieval_agent.agent.datasets['delta_state'].flattened_obs_matrix
        self.delta_s_size = self.delta_s_dataset.shape[-1]
        self.delta_s_scaler = self.retrieval_agent.agent.datasets['delta_state'].obs_scaler
        self.combined_dataset = torch.cat([self.s_dataset, self.a_dataset, self.delta_s_dataset], dim=-1).contiguous()
        self.neighbor_batch = policy_cfg.get("neighbor_batch", -1)
        self.is_diffusion = kwargs.get("diffusion", False)
        self.horizon_mapping = None

        # For ablations, not recommended to stray from defaults
        self.obs_horizon = getattr(self, "obs_horizon", 1)
        self.act_horizon = getattr(self, "act_horizon", 1)

        # Will be set in factory if needed
        self.use_set_transformer = False

        # Validation flags etc.
        self.validation = False
        self.val_start_index = -1
        self.val_delta_s_dataset = []

    def prepare_to_train(self, data_loader):
        if self.validation:
            self.val_start_index = torch.tensor(list(self.retrieval_agent.cache.keys())).max().item() + 1
            index_offset = self.val_start_index
            all_indices = []
        else:
            index_offset = 0

        batch_sampler = data_loader.batch_sampler
        dataset = data_loader.dataset

        for batch_indices in batch_sampler:
            input = torch.stack([dataset[i][0] for i in batch_indices])
            indices = list(batch_indices)

            if self.retrieval_agent.lookback == 1:
                input = input.unsqueeze(dim=1)
            if self.mixed == False:
                input = input.repeat(1, 1, 2)
            input = input.to(self.retrieval_agent.agent.device)
            
            if self.validation:
                all_indices.extend(indices)

            assert input.shape[2] == self.input_splits[-1]

            retrieval_state = input[:, :, 0:self.input_splits[0]]
            delta_state = input[:, -1, self.input_splits[0]:self.input_splits[1]]

            for i in range(len(indices)):
                indices[i] += index_offset

            if self.obs_horizon == 1 and self.act_horizon > 1:
                pass

            self.retrieval_agent.cache_result_for_train(retrieval_state, indices)

            if self.validation:
                self.val_delta_s_dataset.extend(delta_state)

        # We need to build the validation delta s dataset
        if self.validation:
            stacked_dataset = torch.stack(self.val_delta_s_dataset)
            self.val_delta_s_dataset = torch.empty_like(stacked_dataset)

            # Then reorder
            self.val_delta_s_dataset[torch.tensor(all_indices)] = stacked_dataset
            self.val_delta_s_dataset = self.delta_s_scaler.transform(self.val_delta_s_dataset)

    def create_horizon_mapping(self, dataset):
        self.horizon_mapping = torch.empty(len(dataset), self.obs_horizon, device=self.device, dtype=torch.int32)
        for i, indices in enumerate(dataset):
            self.horizon_mapping[i] = indices[0].flatten()

    def forward(self, input):
        index_offset = self.val_start_index if self.validation else 0

        if self.is_diffusion and (self.wrapped.training or self.validation):
            # Each input in batch will be index + action, so split those up
            real_actions = input[:, -self.action_size * self.act_horizon:]

            # Extract indices back to a list of ints
            input = input[:, :self.obs_horizon].to(torch.uint32).squeeze().tolist()

        batch_size = len(input)
        if not (self.wrapped.training or self.validation) and self.retrieval_agent.lookback == 1:
            input = input.unsqueeze(dim=1)
        if self.mixed == False and not (self.wrapped.training or self.validation):
            input = input.repeat(1, 1, 2)

        all_neighbors = []
        all_delta_state = []
        if self.wrapped.training or self.validation:
            for i in input:
                if isinstance(i, list):
                    neighbors = self.retrieval_agent.cache[i[-1] + index_offset]
                else:
                    neighbors = self.retrieval_agent.cache[i + index_offset]

                all_neighbors.append(neighbors)
                if self.obs_horizon > 1:
                    indices = i
                    for index in indices:
                        if self.validation:
                            all_delta_state.append(self.val_delta_s_dataset[index])
                        else:
                            all_delta_state.append(self.delta_s_dataset[index])
                else:
                    if self.validation:
                        all_delta_state.append(self.val_delta_s_dataset[i])
                    else:
                        all_delta_state.append(self.delta_s_dataset[i])
        else:
            assert input.shape[2] == self.input_splits[-1]

            retrieval_state = input[:, :, 0:self.input_splits[0]]
            delta_state = input[:, -self.obs_horizon:, self.input_splits[0]:self.input_splits[1]]
            if delta_state.shape[1] < self.obs_horizon:
                padding_needed = self.obs_horizon - delta_state.shape[1]
                padding = delta_state[:, 0].unsqueeze(1).repeat(1, padding_needed, 1)
                delta_state = torch.cat((padding, delta_state), dim=1)

            delta_state = delta_state.reshape(len(input) * self.obs_horizon, -1)

            neighbors = self.retrieval_agent.get_neighbors(retrieval_state)

            all_neighbors.extend(neighbors)

            all_delta_state.extend(delta_state)

        all_neighbors = torch.stack(all_neighbors)

        if self.obs_horizon > 1:
            all_neighbors = self.horizon_mapping[all_neighbors]
            
        all_neighbors = all_neighbors.flatten()

        # [B * H * k, o + a + o]
        all_data = self.combined_dataset[all_neighbors]

        delta_s_rhs = torch.stack(all_delta_state)
        assert delta_s_rhs.shape
        if not (self.wrapped.training or self.validation):
            delta_s_rhs = self.delta_s_scaler.transform(delta_s_rhs)

        if self.obs_horizon > 1 or self.act_horizon > 1:
            delta_s_rhs = delta_s_rhs.reshape(len(input), self.obs_horizon, -1).repeat(1, self.num_neighbors, 1).reshape(-1, all_delta_state[0].shape[-1])
        else:
            delta_s_rhs = delta_s_rhs.unsqueeze(dim=1).expand(len(input), self.retrieval_agent.num_neighbors, -1).reshape(-1, all_delta_state[0].shape[-1])


        all_data[:, -self.delta_s_size:] -= delta_s_rhs

        # [B, K, D_o] -> [B * K, D_o]
        inputs = all_data.view((-1, all_data.shape[-1] * self.obs_horizon))


        if self.is_diffusion and (self.wrapped.training or self.validation):
            real_actions = real_actions.unsqueeze(dim=1).expand(len(input), self.retrieval_agent.num_neighbors, -1).reshape(-1, self.action_size * self.act_horizon)
            inputs = torch.cat((inputs, real_actions), dim=1)

            noise_loss = self.wrapped(inputs)
            return noise_loss

        all_actions = self.wrapped(inputs)

        if self.use_set_transformer:
            if not (self.wrapped.training or self.validation):
                all_actions = self.output_scaler.transform(all_actions)

            combined_actions = self.set_transformer(all_actions)

            if not (self.wrapped.training or self.validation):
                combined_actions = self.output_scaler.inverse_transform(combined_actions)

            return combined_actions
        else:
            all_actions = all_actions.view(batch_size, self.retrieval_agent.num_neighbors, -1)

            return torch.mean(all_actions, axis=1)

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)

        self.retrieval_agent.agent.to_device(self.device)

        return result

    def compile(self):
        was_training = self.wrapped.training
        if self.is_diffusion:
            # Just compile unet
            self.wrapped.wrapped.model = torch.compile(self.wrapped.wrapped.model, mode="reduce-overhead")
            self.wrapped.wrapped.model._orig_mod.train(was_training)
            object.__setattr__(self.wrapped.wrapped.model, 'training', was_training)
        else:
            self.wrapped = torch.compile(self.wrapped, mode="reduce-overhead")
            self.wrapped._orig_mod.train(was_training)
        object.__setattr__(self.wrapped, 'training', was_training)
        if hasattr(self, "set_transformer"):
            self.set_transformer = torch.compile(self.set_transformer, mode="reduce-overhead")
