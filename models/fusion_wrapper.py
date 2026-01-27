import torch
import torch.nn as nn
from fast_scaler import FastScaler, IdentityScaler, combine_scalers
from logging_util import logger
from models.identity import Identity
from models.model_wrapper import ModelWrapper
from models.scale_wrapper import ScaleWrapper

class FusionWrapper(ModelWrapper):
    def __init__(self, models: list[nn.Module], combination_model: "ModelFactory"):
        super(FusionWrapper, self).__init__()
        input_lens = [0]
        output_lens = [0]
        self.models = nn.ModuleList(models)

        all_identity = True
        all_scale_identity = True
        for model in models:
            all_identity = all_identity and isinstance(model, Identity)
            all_scale_identity = all_scale_identity and (isinstance(model, Identity) or (isinstance(model, ScaleWrapper) and isinstance(model.wrapped, Identity)))
            input_lens.append(model.input_len)
            output_lens.append(model.output_len)

        logger.debug(f"{all_identity=}")
        logger.debug(f"{all_scale_identity=}")
        self.all_identity = all_identity
        self.all_scale_identity = all_scale_identity


        logger.debug(f"{input_lens}")
        logger.debug(f"{output_lens}")
        self.input_splits = torch.cumsum(torch.tensor(input_lens), dim=0)
        self.output_splits = torch.cumsum(torch.tensor(output_lens), dim=0)
        logger.debug(f"{self.input_splits}")
        logger.debug(f"{self.output_splits}")

        combination_model.model_config.input_len = self.output_splits[-1]
        self.combination_model = combination_model.create()

        # All either Identity or scaled Identity, so we can be smart about this
        if self.all_scale_identity:
            scaler = FastScaler()
            for model in models:
                if isinstance(model, Identity):
                    scaler = combine_scalers(scaler, IdentityScaler(model.input_len, torch.float32, self.combination_model.device))
                else:
                    assert isinstance(model, ScaleWrapper)
                    if scaler.mean_torch is not None:
                        scaler = combine_scalers(scaler, model.input_scaler)
                    else:
                        scaler = model.input_scaler
            self.scaler = scaler
        elif not self.all_identity:
            self.intermediate_outputs = torch.empty(
                (0, self.output_splits[-1]),
                device=self.combination_model.device,
                memory_format=torch.contiguous_format
            )

    def forward(self, input):
        if input.shape[1] != self.input_splits[-1]:
            logger.warning("Unexpected input shape in FusionWrapper!")

        if self.all_identity:
            return self.combination_model(input)
        elif self.all_scale_identity:
            return self.combination_model(self.scaler.transform(input))
        else:
            batch_size = len(input)
            if self.intermediate_outputs.shape[0] < batch_size:
                self.intermediate_outputs = torch.empty(
                    (batch_size, self.output_splits[-1]), 
                    device=self.combination_model.device,
                    memory_format=torch.contiguous_format
                )

            for i, model in enumerate(self.models):
                input_start = self.input_splits[i]
                input_end = self.input_splits[i+1]
                output_start = self.output_splits[i]
                output_end = self.output_splits[i+1]

                input_split = input[:, input_start:input_end]
                self.intermediate_outputs[:batch_size, output_start:output_end].copy_(model(input_split))

            return self.combination_model(self.intermediate_outputs[:batch_size])
