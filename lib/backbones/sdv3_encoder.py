import torch
import torch.nn as nn
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
from diffusers import AutoencoderKL
from einops import rearrange, repeat
from ldm.modules.diffusionmodules.util import timestep_embedding


class SDV3Encoder(nn.Module):
    def __init__(self, class_embeddings_path: None):
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
        del pipe.vae.decoder
        self.transformer = pipe.transformer
        self.class_embedding = torch.load(class_embeddings_path)
        self.text_adapter = TextAdapterDepth(text_dim=768)
        self.gamma = nn.Parameter(torch.ones(768) * 1e-4)

    def forward(self, x):
        with torch.no_grad():
            latents = self.vae.encode(x, return_dict=False)[0].mode().detach()

        c_crossattn = self.text_adapter(latents, self.class_embedding, self.gamma)
        c_crossattn = c_crossattn.repeat(x.shape[0], 1, 1)

        # timestep embedding
        t = torch.ones((x.shape[0]), device=x.device).long()

        # finetune the backbone
        feats = self.transformer(hidden_states=latents, encoder_hidden_states=c_crossattn, timestep=t)
        return feats


class TextAdapterDepth(nn.Module):
    def __init__(self, text_dim=768):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, text_dim)
        )

    def forward(self, latents, texts, gamma):
        # use the gamma to blend
        gamma = gamma.to(latents.device)
        texts = texts[0].unsqueeze(0)
        n_sen, channel = texts.shape
        bs = latents.shape[0]

        texts_after = self.fc(texts)
        texts = texts + gamma * texts_after
        texts = repeat(texts, 'n c -> n b c', b=1)
        return texts


# main, for test purposes
if __name__ == '__main__':
    model = SDV3Encoder(class_embeddings_path='/mnt/nodestor/MDP/kitti_embeddings.pth')
    model.to('cuda')
    x = torch.randn(1, 3, 224, 224).to('cuda')
    print(model)
    print(model(x).shape)
