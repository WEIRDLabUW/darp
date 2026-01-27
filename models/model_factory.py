import math
from types import SimpleNamespace

import torch.nn as nn

from constants import RESNET_SIZE
from models.darp_wrapper import DARPWrapper
from models.diffusion import DiffusionPolicy
from models.fusion_wrapper import FusionWrapper
from models.gaussian import GMMWrapper
from models.identity import Identity
from models.mlp import MLP
from models.model_utils import get_min_max_len, get_io_size_from_data_path, get_scalers_from_data_path, set_attributes_from_args
from models.r3m import R3M
from models.scale_wrapper import DiffusionScaleWrapper, ScaleWrapper
import copy

from models.set_transformer import SetTransformer

class ModelFactory():
    def __init__(self, config: dict, **kwargs):
        self.model_config = SimpleNamespace()

        print("Setting attributes in ModelFactory")
        set_attributes_from_args(self.model_config, config, kwargs)

        # Flags that change construction
        self.classifier = getattr(self.model_config, "classifier", False)
        self.darp = getattr(self.model_config, "darp", False)
        self.tune = getattr(self.model_config, "tune", False)
        self.sideload_r3m = getattr(self.model_config, "sideload_r3m", False)
        self.sideload_set_transformer = getattr(self.model_config, "sideload_set_transformer", False)
        self.sideload_set_transformer_gmm = getattr(self.model_config, "sideload_set_transformer_gmm", False)

    def create(self) -> nn.Module:
        model = None
        if self.model_config.type == "r3m_fusion":
            intermediate_models = []
            num_viewpoints = len(self.model_config.env_cfg['cams'])

            assert hasattr(self.model_config, "prop_config")
            input_len, output_len = get_io_size_from_data_path(self.model_config.demo_pkl, classifier=self.classifier)
            if not self.tune:
                input_len -= num_viewpoints * RESNET_SIZE
            prop_config = self.model_config.prop_config
            prop_config['device'] = self.model_config.device
            prop_config['demo_pkl'] = self.model_config.env_cfg['prop_demo_pkl']
            prop_config['input_len'] = input_len
            intermediate_models.append(ModelFactory(prop_config).create())

            assert hasattr(self.model_config, "r3m_config")
            r3m_config = self.model_config.r3m_config
            r3m_config['device'] = self.model_config.device

            if self.tune:
                r3m_config['input_len'] = self.model_config.r3m_config['rgb_height'] * self.model_config.r3m_config['rgb_width'] * 3
            else:
                r3m_config['input_len'] = RESNET_SIZE

            for _ in range(num_viewpoints):
                if self.tune:
                    inner_config = copy.deepcopy(r3m_config)
                    inner_config['input_len'] = RESNET_SIZE

                    inner_model = ModelFactory(inner_config).create()

                    r3m_config['optimizer'] = r3m_config['tune_optimizer']
                    intermediate_models.append(R3M(inner_model, **r3m_config))
                else:
                    intermediate_models.append(ModelFactory(r3m_config).create())

            assert hasattr(self.model_config, "fusion_config")
            fusion_config = self.model_config.fusion_config
            fusion_config['device'] = self.model_config.device
            fusion_config['output_len'] = output_len
            fusion_config['demo_pkl'] = self.model_config.demo_pkl

            model = FusionWrapper(intermediate_models, ModelFactory(fusion_config))

            if not self.tune:
                # We aren't fine-tuning R3M but still need it at inference-time
                # Side-load but don't backprop through weights
                dummy_module = nn.Module()
                dummy_module.output_len = 0
                model.r3m = R3M(dummy_module, **r3m_config)
                model.r3m.eval()
                for param in model.r3m.parameters():
                    param.requires_grad = False
        elif self.model_config.type == "r3m":
            model = R3M(**vars(self.model_config))
        elif self.model_config.type == "identity":
            model = Identity(**vars(self.model_config))
        elif self.model_config.type == "gmm":
            self.model_config.type = self.model_config.wrapped_type
            if getattr(self.model_config, 'assume_io_size', False):
                self.model_config.input_len, self.model_config.output_len = get_io_size_from_data_path(self.model_config.demo_pkl, classifier=self.classifier, darp=self.darp)
            self.model_config.output_len *= self.model_config.num_modes * 2
            self.model_config.output_len += self.model_config.num_modes
            self.model_config.assume_io_size = False

            scale = self.model_config.scale
            darp = getattr(self.model_config, "darp", False)
            self.model_config.scale = False
            self.model_config.darp = False
            wrapped_model = ModelFactory(vars(self.model_config))
            model = GMMWrapper(wrapped_model.create(), self.model_config.num_modes)
            self.model_config.scale = scale
            self.model_config.darp = darp
        elif self.model_config.type == "diffusion":
            if getattr(self.model_config, 'assume_io_size', False):
                assert hasattr(self.model_config, "demo_pkl"), "To assume io size, include attribute 'demo_pkl' in your model config file that point to the data you'd like to assume io size from."
                self.model_config.input_len, self.model_config.output_len = get_io_size_from_data_path(self.model_config.demo_pkl, classifier=self.classifier, darp=self.darp)

            self.model_config.input_len += self.model_config.output_len

            model = DiffusionPolicy(**vars(self.model_config))
        elif self.model_config.type == "r3m_darp":
            intermediate_models = []
            num_viewpoints = len(self.model_config.env_cfg['cams'])
            input_len, output_len = get_io_size_from_data_path(self.model_config.demo_pkl, classifier=self.classifier)

            assert hasattr(self.model_config, "prop_config")
            input_len -= num_viewpoints * RESNET_SIZE
            prop_config = self.model_config.prop_config
            prop_config['device'] = self.model_config.device
            prop_config['demo_pkl'] = self.model_config.env_cfg['prop_demo_pkl']
            prop_config['input_len'] = input_len

            assert hasattr(self.model_config, "act_config")
            act_config = self.model_config.act_config
            act_config['device'] = self.model_config.device
            act_config['demo_pkl'] = self.model_config.env_cfg['act_demo_pkl']
            act_config['input_len'] = output_len

            assert hasattr(self.model_config, "r3m_config")
            r3m_config = self.model_config.r3m_config
            r3m_config['device'] = self.model_config.device
            r3m_config['input_len'] = RESNET_SIZE

            # s
            intermediate_models.append(ModelFactory(prop_config).create())

            for _ in range(num_viewpoints):
                intermediate_models.append(ModelFactory(r3m_config).create())

            # a
            intermediate_models.append(ModelFactory(act_config).create())

            # delta s
            intermediate_models.append(ModelFactory(prop_config).create())

            for _ in range(num_viewpoints):
                intermediate_models.append(ModelFactory(r3m_config).create())

            assert hasattr(self.model_config, "fusion_config")
            fusion_config = self.model_config.fusion_config
            fusion_config['device'] = self.model_config.device
            fusion_config['output_len'] = output_len
            fusion_config['demo_pkl'] = self.model_config.demo_pkl

            model = FusionWrapper(intermediate_models, ModelFactory(fusion_config))
        else:
            if not self.model_config.type == "mlp":
                print(f"Model type {self.model_config.type} is not supported! Defaulting to MLP")

            if getattr(self.model_config, 'assume_io_size', False):
                assert hasattr(self.model_config, "demo_pkl"), "To assume io size, include attribute 'demo_pkl' in your model config file that point to the data you'd like to assume io size from."
                ret_config = getattr(self.model_config, "retrieval_config", {})
                self.model_config.input_len, self.model_config.output_len = get_io_size_from_data_path(self.model_config.demo_pkl, classifier=self.classifier, darp=self.darp)

            model = MLP(**vars(self.model_config))

        if self.sideload_r3m:
            # We aren't fine-tuning R3M but still need it at inference-time
            # Side-load but don't backprop through weights
            r3m_config = self.model_config.r3m_config
            r3m_config['device'] = self.model_config.device

            dummy_module = nn.Module()
            dummy_module.output_len = 0

            model.r3m = R3M(dummy_module, **r3m_config)
            model.r3m.eval()

            for param in model.r3m.parameters():
                param.requires_grad = False


        if getattr(self.model_config, "scale", False):
            # Need to have path to the data to fit scalers
            assert hasattr(self.model_config, "demo_pkl"), "To scale, include attribute 'demo_pkl' in your model config file that point to the data you'd like to fit the scalers to."

            input_scaler, output_scaler = get_scalers_from_data_path(self.model_config.demo_pkl, darp=self.darp)

            if self.model_config.type == "diffusion":
                if not getattr(self.model_config, "scale_input", True):
                    input_scaler = None

                act_min, act_max, act_len, obs_min, obs_max, obs_len = get_min_max_len(self.model_config.demo_pkl, norm_obs=self.darp)
                model = DiffusionScaleWrapper(model, obs_min, obs_max, obs_len, act_min, act_max, act_len, getattr(self.model_config, "obs_horizon", 1), getattr(self.model_config, "act_horizon", 1), darp=self.darp)
            else:
                if not getattr(self.model_config, "scale_input", True):
                    input_scaler = None

                if not getattr(self.model_config, "scale_output", True):
                    output_scaler = None

                model = ScaleWrapper(model, input_scaler, output_scaler)
                model = model.to(self.model_config.device)

        if self.darp:
            model = DARPWrapper(model, self.model_config.env_cfg, self.model_config.retrieval_config, diffusion=self.model_config.type == "diffusion")

        if self.sideload_set_transformer:
            set_transformer_config = self.model_config.set_transformer_config
            set_transformer_config['device'] = self.model_config.device

            assert hasattr(self.model_config, 'retrieval_config')
            ret_config = self.model_config.retrieval_config
            set_transformer_config['set_len'] = math.floor(ret_config['k'] * ret_config['final_neighbors_ratio'])

            set_transformer_config['output_len'] = model.output_len
            set_transformer_config['input_len'] = set_transformer_config['output_len']
            model.set_transformer = SetTransformer(**set_transformer_config)
            model.use_set_transformer = True

        if self.sideload_set_transformer_gmm:
            set_transformer_config = self.model_config.set_transformer_config
            set_transformer_config['device'] = self.model_config.device

            assert hasattr(self.model_config, 'retrieval_config')
            ret_config = self.model_config.retrieval_config
            set_transformer_config['set_len'] = math.floor(ret_config['k'] * ret_config['final_neighbors_ratio'])

            set_transformer_config['output_len'] = (model.output_len * self.model_config.num_modes * 2) + self.model_config.num_modes
            set_transformer_config['input_len'] = self.model_config.output_len
            wrapped_transformer = SetTransformer(**set_transformer_config)
            model.set_transformer = GMMWrapper(wrapped_transformer, self.model_config.num_modes)
            model.use_set_transformer = True

        return model
