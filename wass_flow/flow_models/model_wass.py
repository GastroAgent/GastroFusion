import glob
import json
import math
from tqdm import tqdm
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import random
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from flow_matcher import create_generator # 不包含 x0, x1.
from utils.data_loader_test import MedicalJsonDataset
from utils.data_utils_test import create_dataloaders_by_pairs
from utils.train_utils import infiniteloop
from PIL import Image
from math import sqrt
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from einops.layers.torch import Rearrange

class EmbeddingNetwork(nn.Module):
    def __init__(self, pretrained=False, freeze_base=False, model='resnet34'):
        super(EmbeddingNetwork, self).__init__()
        if model == 'resnet34':
            self.base_model = models.resnet34(pretrained=pretrained)
        else:
            self.base_model = models.resnet101(pretrained=pretrained)
        
        self.base_model.conv1 = nn.Conv2d(4, self.base_model.conv1.out_channels, kernel_size=3, stride=1, padding=1,
                                          bias=False)
        # 修改输出为 256 维特征向量
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 1024)
        
        # 可选：冻结预训练模型参数
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # L2 归一化
        self.l2norm = nn.functional.normalize

    def forward(self, x, return_hidden=False, **kwargs):
        features = self.base_model(x)
        if return_hidden:
            return self.l2norm(features, p=2, dim=1), features
        else:
            features = self.l2norm(features, p=2, dim=1)  # L2 归一化
            return features

class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=4, in_chans=4, embed_dim=64):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, 4, 64, 64]
        x = self.proj(x)  # -> [B, embed_dim, 8, 8]
        x = x.flatten(2)  # [B, embed_dim, 64]
        x = x.transpose(1, 2)  # [B, 64, embed_dim]
        return x  # [B, num_patches, embed_dim]

class SimpleTransformer(nn.Module):
    def __init__(self, embed_dim=64, depth=2, num_heads=4, mlp_ratio=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class TanhTransformer(nn.Module):
    def __init__(self, embed_dim=64, depth=2, num_heads=4, mlp_ratio=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            convert_ln_to_dyt(nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.norm = convert_ln_to_dyt(self.norm)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

import torch
import torch.nn as nn
from timm.layers import LayerNorm2d

class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = 1e-6
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"

def convert_ln_to_dyt(module):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = DynamicTanh(module.normalized_shape, not isinstance(module, LayerNorm2d))
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_dyt(child))
    del module
    return module_output

class UpsampleDecoder(nn.Module):
    def __init__(self, embed_dim=64, patch_size=8, out_chans=4):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # 将嵌入还原为特征图
        self.proj = GatedMLPClassifier(embed_dim, embed_dim, patch_size * patch_size * out_chans, patch_size * patch_size * out_chans)
        self.linear = nn.Linear(patch_size * patch_size * out_chans, 4)
    def forward(self, x):
        # x: [B, num_patches, embed_dim]
        x = self.proj(x)  # [B, num_patches, 8*8*4]
        # 重塑为 [B, C, H, W]
        B, N, _ = x.shape
        h = w = int(N ** 0.5)  # 假设是正方形
        x = x.reshape(B, h, w, -1)
        x = self.linear(x)
        return x.permute(0,3,1,2)  # [B, 16, 16, 4]

class AttentionDownEncoderXL(nn.Module):
    def __init__(self, dy=False):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=64, patch_size=4, in_chans=4, embed_dim=512)
        if dy:
            self.transformer = TanhTransformer(embed_dim=512, depth=12, num_heads=8, mlp_ratio=4)
        else:
            self.transformer = SimpleTransformer(embed_dim=512, depth=12, num_heads=8, mlp_ratio=4)
            
        self.decoder = UpsampleDecoder(embed_dim=512, patch_size=4, out_chans=4)

    def forward(self, x, return_hidden=False):
        # x: [B, 4, 64, 64]
        x = self.patch_embed(x)        # -> [B, 256, 512]
        hidden_states = self.transformer(x)        # -> [B, 256, 512]
        output = self.decoder(hidden_states)            # -> [B, 4, 64, 64]
        B = output.shape[0]
        if return_hidden:
            return output.reshape(B, -1), hidden_states
        return output.reshape(B, -1)

class GatedMLPClassifier(nn.Module):
    def __init__(self, input_dim=4 * 64 * 64, hidden_dim1=4096, hidden_dim2=1024, output_dim=256):
        super().__init__()
        # 第一层：门控 MLP（SwiGLU 风格）
        self.gate1 = nn.Linear(input_dim, hidden_dim1 * 2)  # 输出拼接 [gate | value]
        # 第二层
        self.gate2 = nn.Linear(hidden_dim1, hidden_dim2 * 2)
        # 输出层（可选是否门控；这里简化为普通线性）
        self.out_proj = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        # Layer 1
        gate_val = self.gate1(x)
        gate, val = gate_val.chunk(2, dim=-1)  # split into two halves
        x = nn.functional.silu(gate) * val     # SwiGLU: SiLU(gate) ⊙ value

        # Layer 2
        gate_val = self.gate2(x)
        gate, val = gate_val.chunk(2, dim=-1)
        x = nn.functional.silu(gate) * val

        # Output projection
        x = self.out_proj(x)
        return x

class TripletNetwork(nn.Module):
    def __init__(self, pretrained=False, freeze_base=False, model='resnet34', dy=False):
        super(TripletNetwork, self).__init__()
        if model == 'attention':
            self.embedding = AttentionDownEncoderXL(dy)
        else:
            self.embedding = EmbeddingNetwork(pretrained=pretrained, freeze_base=freeze_base, model=model)

    def encode(self, x, return_image=False, return_hidden=False):
        embeds = self.embedding(x, return_hidden)
        if return_hidden:
            hidden_state = embeds[1]
            embeds = embeds[0]
            
        if embeds.ndim == 2:
            B, D = embeds.shape
            if return_image:
                H = W = int(sqrt(D // 4))
                embeds = embeds.view(B, 4, H, W)

        elif embeds.ndim == 4 :
            B, C, H, W = embeds.shape
            if return_image:
                embeds = embeds 
            else:
                embeds = embeds.view(B, -1)
        
        if return_hidden:
            return embeds, hidden_state
        else:
            return embeds
        
    def forward(self, anchor, positive, negative):
        anchor_emb = self.encode(anchor, True)
        positive_emb = self.encode(positive, True)
        negative_emb = self.encode(negative, True)
        return anchor_emb, positive_emb, negative_emb