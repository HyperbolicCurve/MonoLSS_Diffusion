from diffusers import StableDiffusion3Pipeline
import torch

if __name__ == '__main__':
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
    ).to("cuda")
    # kitti dataset
    prompt_1 = "A photo shows cars on city road."
    prompt_2 = "A photo shows cars on city road, and the cars are parked on the side"
    prompt_3 = "A photo shows cars on city road, there may be some pedestrians and cyclist on the road"
    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(prompt_1, prompt_2, prompt_3)

    # save prompt embeddings and pooled prompt embeddings as .pth files
    torch.save(prompt_embeds, 'prompt_embeds.pth')
    torch.save(pooled_prompt_embeds, 'pooled_prompt_embeds.pth')