import math
import torch
import torch.nn as nn
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
from diffusers import AutoencoderKL
from einops import rearrange, repeat
from ldm.modules.diffusionmodules.util import timestep_embedding


class SDV3Encoder(nn.Module):
    def __init__(self, class_embedding_path, pooled_projection_path):
        super(SDV3Encoder, self).__init__()
        torch.set_float32_matmul_precision("high")
        torch._inductor.config.conv_1x1_as_mm = True
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.epilogue_fusion = False
        torch._inductor.config.coordinate_descent_check_all_directions = True
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
        ).to("cuda")
        pipe.transformer.to(memory_format=torch.channels_last)
        pipe.vae.to(memory_format=torch.channels_last)
        self.vae = pipe.vae
        self.vae.requires_grad_(False)  # freeze the weights of the VAE
        del pipe.vae.decoder
        self.transformer = pipe.transformer
        self.class_embedding = torch.load(class_embedding_path)
        self.pooled_projection = torch.load(pooled_projection_path)
        self.gamma = nn.Parameter(torch.ones(768) * 1e-4)

    def forward(self, x):
        with torch.no_grad():
            latents = self.vae.encode(x, return_dict=False)[0].mode().detach()

        # timestep embedding
        t = torch.ones((x.shape[0]), device=x.device).long()

        # finetune the backbone
        feats = self.transformer(hidden_states=latents, encoder_hidden_states=self.class_embedding, timestep=t,
                                 pooled_projections=self.pooled_projection, return_dict=False)
        return feats

