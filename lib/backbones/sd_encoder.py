"""
VPD dependencies
"""
from omegaconf import OmegaConf
import torch as th
import torch
import math
import abc
from torch import nn, einsum
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_
from inspect import isfunction
import torch.nn.functional as F
from ldm.util import instantiate_from_config
from typing import Dict, List


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class VPDEncoder(nn.Module):
    def __init__(self,
                 out_dim=1024,
                 ldm_prior=[320, 640, 1280 + 1280],
                 sd_path=None,
                 text_dim=768,
                 train_backbone=False,
                 return_interm_layers=True,
                 class_embeddings_path=None,
                 sd_config_path=None,
                 sd_checkpoint_path=None,
                 use_attn=False):
        super().__init__()
        if return_interm_layers:
            if use_attn == False:
                self.strides = [8, 16, 32]
                self.num_channels = [320, 640, 2560]
            else:
                self.strides = [8, 16, 32]
                self.num_channels = [320, 641, 2561]
        else:
            self.strides = [32]
            self.num_channels = [2560]
        self.train_backbone = train_backbone
        self.layer1 = nn.Sequential(
            nn.Conv2d(ldm_prior[0], ldm_prior[0], 3, stride=2, padding=1),
            nn.GroupNorm(16, ldm_prior[0]),
            nn.ReLU(),
            nn.Conv2d(ldm_prior[0], ldm_prior[0], 3, stride=2, padding=1),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(ldm_prior[1], ldm_prior[1], 3, stride=2, padding=1),
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(sum(ldm_prior), out_dim, 1),
            nn.GroupNorm(16, out_dim),
            nn.ReLU(),
        )

        self.apply(self._init_weights)

        ### stable diffusion layers

        config = OmegaConf.load(sd_config_path)
        config.model.params.ckpt_path = sd_checkpoint_path

        sd_model = instantiate_from_config(config.model)
        self.encoder_vq = sd_model.first_stage_model
        # self.encoder_vq = torch.compile(encoder_vq, mode='max-autotune', fullgraph=True)
        self.unet = UNetWrapper(sd_model.model, use_attn=use_attn)

        del sd_model.cond_stage_model
        del self.encoder_vq.decoder
        del self.unet.unet.diffusion_model.out

        for param in self.encoder_vq.parameters():
            param.requires_grad = False

        self.text_adapter = TextAdapterDepth(text_dim=text_dim)
        self.class_embeddings = torch.load(class_embeddings_path)
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, feats):
        x = self.ldm_to_net[0](feats[0])
        for i in range(3):
            if i > 0:
                x = x + self.ldm_to_net[i](feats[i])
            x = self.layers[i](x)
            x = self.upsample_layers[i](x)
        return self.out_conv(x)

    def forward(self, x, class_ids=None):
        with torch.no_grad():
            latents = self.encoder_vq.encode(x).mode().detach()
        class_embeddings = []
        for class_embedding in self.class_embeddings:
            class_embeddings.append(class_embedding.to(latents.device))

        c_crossattn = self.text_adapter(latents, class_embeddings,
                                        self.gamma)  # NOTE: here the c_crossattn should be expand_dim as latents
        c_crossattn = c_crossattn.repeat(x.shape[0], 1, 1)
        t = torch.ones((x.shape[0],), device=x.device).long()
        # import pdb; pdb.set_trace()
        if self.train_backbone:
            outs = self.unet(latents, t, c_crossattn=[c_crossattn])
        else:
            with torch.no_grad():
                outs = self.unet(latents, t, c_crossattn=[c_crossattn])
        feats = [outs[0], outs[1], torch.cat([outs[2], F.interpolate(outs[3], scale_factor=2)], dim=1)]
        # feats_upsampled = [F.interpolate(feat, scale_factor=2) for feat in feats]
        # feats_original = [feat[:,:,feat.shape[2]//2-int(round(feat.shape[3]*384/1280/2)):feat.shape[2]//2+int(round(feat.shape[3]*384/1280/2)),:] for feat in feats_upsampled]
        # out = {}
        # for name, x in enumerate(feats):
        #     m = torch.zeros(x.shape[0], x.shape[2], x.shape[3]).to(torch.bool).to(x.device)
        #     out[f"{name}"] = NestedTensor(x, m)
        # 8 48*160 16 24*80 32 12*40
        # x = torch.cat([self.layer1(feats[0]), self.layer2(feats[1]), feats[2]], dim=1)
        # out = self.out_layer(x)
        return feats


class UNetWrapper(nn.Module):
    def __init__(self, unet, use_attn=True, base_size=1280, max_attn_size=None,
                 attn_selector='up_cross+down_cross') -> None:
        super().__init__()
        self.unet = unet
        self.attention_store = AttentionStore(base_size=base_size // 8, max_size=max_attn_size)
        self.size16 = base_size // 32
        self.size32 = base_size // 16
        self.size64 = base_size // 8
        self.use_attn = use_attn
        if self.use_attn:
            register_attention_control(unet, self.attention_store)
        register_hier_output(unet)
        self.attn_selector = attn_selector.split('+')

    def forward(self, *args, **kwargs):
        if self.use_attn:
            self.attention_store.reset()
        out_list = self.unet(*args, **kwargs)
        if self.use_attn:
            avg_attn = self.attention_store.get_average_attention()
            attn16, attn32, attn64 = self.process_attn(avg_attn)
            out_list[1] = torch.cat([out_list[1], attn16], dim=1)
            out_list[2] = torch.cat([out_list[2], attn32], dim=1)
            if attn64 is not None:
                out_list[3] = torch.cat([out_list[3], attn64], dim=1)
        return out_list[::-1]

    def process_attn(self, avg_attn):
        attns = {self.size16: [], self.size32: [], self.size64: []}
        for k in self.attn_selector:
            for up_attn in avg_attn[k]:
                size = int(round(math.sqrt(up_attn.shape[1] * 1280 / 384)))
                attns[size].append(rearrange(up_attn, 'b (h w) c -> b c h w', w=size))
        attn16 = torch.stack(attns[self.size16]).mean(0)
        attn32 = torch.stack(attns[self.size32]).mean(0)
        if len(attns[self.size64]) > 0:
            attn64 = torch.stack(attns[self.size64]).mean(0)
        else:
            attn64 = None
        return attn16, attn32, attn64


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


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        attn = self.forward(attn, is_cross, place_in_unet)
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= (self.max_size) ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item for item in self.step_store[key]] for key in self.step_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, base_size=64, max_size=None):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.base_size = base_size
        if max_size is None:
            self.max_size = self.base_size // 2
        else:
            self.max_size = max_size


def register_hier_output(model):
    self = model.diffusion_model
    from ldm.modules.diffusionmodules.util import checkpoint, timestep_embedding
    def forward(x, timesteps=None, context=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
                self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            # import pdb; pdb.set_trace()
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        out_list = []

        for i_out, module in enumerate(self.output_blocks):
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
            if i_out in [1, 4, 7]:
                out_list.append(h)
        h = h.type(x.dtype)

        out_list.append(h)
        return out_list

    self.forward = forward


def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        def forward(x, context=None, mask=None):
            h = self.heads

            q = self.to_q(x)
            is_cross = context is not None
            context = default(context, x)
            k = self.to_k(context)
            v = self.to_v(context)

            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

            if exists(mask):
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)

            attn2 = rearrange(attn, '(b h) k c -> h b k c', h=h).mean(0)
            controller(attn2, is_cross, place_in_unet)

            out = einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
            return self.to_out(out)

        return forward

    class DummyController:
        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.diffusion_model.named_children()

    for net in sub_nets:
        if "input_blocks" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "output_blocks" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "middle_block" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count
