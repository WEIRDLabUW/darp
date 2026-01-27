from nn_agent import NN_METHOD, NNAgentEuclideanStandardized
import torch.nn as nn
import math
import torch
from logging_util import logger
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64

class RetrievalAgent(nn.Module):
    def __init__(self, env_cfg, policy_cfg):
        super(RetrievalAgent, self).__init__()

        self.agent = NNAgentEuclideanStandardized(env_cfg, policy_cfg)
        self.num_neighbors = math.floor(self.agent.candidates * self.agent.final_neighbors_ratio)
        self.lookback = self.agent.lookback
    
        logger.info(f"Retrieving with k={self.agent.candidates}, lookback={self.lookback}, decay={self.agent.decay}, ratio={self.agent.final_neighbors_ratio}")

        self.cache = {}

    # Used at training time
    def cache_result_for_train(self, state, indices):
        # Create batch dim
        if state.ndim == 2:
            state = state.unsqueeze(0)

        assert state.shape[0] == len(indices)

        all_neighbors = self.get_neighbors(state)
        for i, index in enumerate(indices):
            if index in self.cache:
                logger.warning(f"Hash collision! {index}")
            else:
                if self.agent.method == NN_METHOD.KNN_AND_DELTA:
                    self.cache[index] = (all_neighbors[0][i], all_neighbors[1][i])
                else:
                    self.cache[index] = all_neighbors[i]

    def get_neighbors(self, input):
        return self.agent.get_neighbors(input)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['cache'] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'cache'):
            self.cache = {}
