# ablangpaired_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from transformers import PreTrainedModel, PretrainedConfig, AutoModel, AutoConfig
from safetensors.torch import load_file

import typing as T


class Mixer(nn.Module):
    def __init__(self, in_d: int=1536):
        super(Mixer, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_d, in_d), # First layer
            nn.ReLU(),             # First activation function
            nn.Linear(in_d, in_d), # Second layer
            nn.ReLU(),             # Second activation function
            nn.Linear(in_d, in_d), # Third layer
            nn.ReLU(),             # Third activation function
            nn.Linear(in_d, in_d), # Fourth layer
            nn.ReLU(),             # Fourth activation function
            nn.Linear(in_d, in_d),  # Fifth layer
            nn.ReLU(),             # Fifth activation function
            nn.Linear(in_d, in_d)      # Output layer
            # No activation here, apply softmax or sigmoid externally if needed, depending on your loss function
        )

    def forward(self, x):
        return self.layers(x)


def get_sequence_embeddings(mask, model_output):
    mask = mask.float()
    d = {k: v for k, v in torch.nonzero(mask).cpu().numpy()} # dict of sep tokens k = ab index, v = index of final position where mask = 1
    # make sep token invisible
    for i in d:
        mask[i, d[i]] = 0
    mask[:, 0] = 0.0 # make cls token invisible
    mask = mask.unsqueeze(-1).expand(model_output.last_hidden_state.size())
    sum_embeddings = torch.sum(model_output.last_hidden_state * mask, 1)
    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
    return sum_embeddings / sum_mask  # sum_mask means length of unmasked positions


class AbLangPairedConfig(PretrainedConfig):
    model_type = "ablang_paired"

    def __init__(
        self,
        checkpoint_filename: str,
        heavy_model_id='qilowoq/AbLang_heavy',
        heavy_revision='ecac793b0493f76590ce26d48f7aac4912de8717',
        light_model_id='qilowoq/AbLang_light',
        light_revision='ce0637166f5e6e271e906d29a8415d9fdc30e377',
        mixer_hidden_dim: int = 1536,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.checkpoint_filename = checkpoint_filename
        self.heavy_model_id = heavy_model_id
        self.heavy_revision = heavy_revision
        self.light_model_id = light_model_id
        self.light_revision = light_revision
        self.mixer_hidden_dim = mixer_hidden_dim


class AbLangPaired(PreTrainedModel):

    def __init__(self, personal_config: AbLangPairedConfig, device: T.Union[str, torch.device] = "cpu"):
        # During training I used the AbLang_heavy config as AbLangPaired's config
        # This may be why it is very hard to integrate this into the Hugging Face AutoModel system
        self.config = AutoConfig.from_pretrained(personal_config.heavy_model_id, revision=personal_config.heavy_revision)
        super().__init__(self.config)
        # super().__init__()

        self.roberta_heavy = AutoModel.from_pretrained(
            personal_config.heavy_model_id,
            revision=personal_config.heavy_revision,  # Specific commit hash
            trust_remote_code=True
        )

        self.roberta_light = AutoModel.from_pretrained(
            personal_config.light_model_id,
            revision=personal_config.light_revision,  # Specific commit hash
            trust_remote_code=True
        )

        self.mixer = Mixer(in_d=1536)

        # Load either torch or transformers saved file
        if personal_config.checkpoint_filename.endswith('.safetensors'):
            state_dict = load_file(personal_config.checkpoint_filename)
        else:
            state_dict = torch.load(personal_config.checkpoint_filename, map_location=device)

        load_result = self.load_state_dict(state_dict, strict=False)
        self.to(device)
        self.eval()

    def forward(self, h_input_ids, h_attention_mask, l_input_ids, l_attention_mask, **kwargs):
        # Run chains through separate streams
        outputs_h = self.roberta_heavy(input_ids=h_input_ids.to(torch.int64), attention_mask=h_attention_mask)
        outputs_l = self.roberta_light(input_ids=l_input_ids.to(torch.int64), attention_mask=l_attention_mask)

        # Mean pool
        pooled_output_h = get_sequence_embeddings(h_attention_mask, outputs_h)
        pooled_output_l = get_sequence_embeddings(l_attention_mask, outputs_l)

        # Concatenate and then do 6 fully connected layers to pick up on cross-chain features
        pooled_output = torch.cat([pooled_output_h, pooled_output_l], dim=1)
        pooled_output = self.mixer(pooled_output)
        embedding = F.normalize(pooled_output, p=2, dim=1)
        return embedding