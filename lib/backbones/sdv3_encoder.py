import torch
import torch.nn as nn
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline



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
        )
        pipe.transformer.to(memory_format=torch.channels_last)
        pipe.vae.to(memory_format=torch.channels_last)

        self.vae = pipe.vae
        self.vae.requires_grad_(False)  # freeze the weights of the VAE
        del pipe.vae.decoder

        self.transformer = pipe.transformer
        self.class_embedding = torch.load(class_embedding_path)
        self.pooled_projection = torch.load(pooled_projection_path)

    def forward(self, x):
        device = x.device
        self.class_embedding = self.class_embedding.to(device)
        self.pooled_projection = self.pooled_projection.to(device)

        with torch.no_grad():
            latents = self.vae.encode(x, return_dict=False)[0].mode().detach()

        latents = latents.to(device)
        # timestep embedding
        t = torch.ones((x.shape[0]), device=device).long()
        # finetune the backbone
        feats = self.transformer(hidden_states=latents, encoder_hidden_states=self.class_embedding, timestep=t,
                                 pooled_projections=self.pooled_projection, return_dict=False)
        return feats
