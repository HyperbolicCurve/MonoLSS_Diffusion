import torch
import torch.nn as nn
from diffusers import SD3Transformer2DModel
from diffusers import AutoencoderKL

class SDV3Encoder(nn.Module):
    def __init__(self,
                 class_embeddings_path: None,
                 ):
        self.autoencoder = AutoencoderKL.from_pretrained('/mnt/nodestor/MonoLSS/weights/vae-ft-mse-840000-ema-pruned'
                                                         '.safetensors')
        self.blocks = SD3Transformer2DModel.from_pretrained('/mnt/nodestor/MonoLSS/weights/sd3_medium.safetensors')
        self.class_embedding = torch.load(class_embeddings_path)
    def forward(self, x):
        with torch.no_grad():
            latents = self.autoencoder.encode(x)
        class_embedding = []
        for class_embedding in self.class_embedding:
            class_embedding.append(class_embedding.to(latents.device))

        c_crossattn = self.text_adapter(latents, class_embedding,
                                        self.gamma)
        c_crossattn =  c_crossattn.repeat(x.shape[0], 1, 1)

        # timestep embedding
        t = torch.ones((x.shape[0]), device=x.device).long()

        # finetune the backbone
        feats = self.backbone(latents, t, c_crossattn = [c_crossattn])


# main, for test purposes
if __name__ == '__main__':
    model = SDV3Encoder(class_embeddings_path='/mnt/nodestor/MDP/kitti_embeddings.pth')
    x = torch.randn(1, 3, 224, 224)
    print(model)
    print(model(x).shape)