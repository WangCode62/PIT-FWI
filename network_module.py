"""Network module for the PIT project.

This file contains convolutional blocks, the Transformer-based decoder,
channel-fusion modules, and the Siamese network used in stage two.
"""

from functools import partial
from typing import Tuple

import torch
import torch.nn as nn


class SubBlock(nn.Module):
    """Basic convolution block: Conv + BN + LeakyReLU."""
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Eblock(nn.Module):
    """Encoder/decoder block.

    Use ``operation='down'`` for downsampling and ``operation='up'`` for upsampling.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        operation: str,
        final_shape: Tuple[int, int] | None = None,
    ):
        super().__init__()
        layers = [
            SubBlock(in_channels=in_channels, out_channels=out_channels, stride=stride),
            SubBlock(in_channels=out_channels, out_channels=out_channels, stride=stride),
        ]
        if operation == "down":
            layers.append(nn.MaxPool2d(kernel_size=2))
        elif operation == "up":
            upsample = nn.Upsample(final_shape, mode="bilinear") if final_shape else nn.Upsample(scale_factor=2, mode="bilinear")
            layers.append(upsample)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Fusion(nn.Module):
    """Two-input feature fusion module kept for notebook compatibility."""
    def __init__(self, in_channels: int):
        super().__init__()
        self.vconv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, stride=1)
        self.fconv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1)

    def forward(self, xp: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        if xp.shape != xs.shape:
            raise ValueError(f"Vp shape {tuple(xp.shape)} does not match Vs shape {tuple(xs.shape)}")
        xp = self.vconv(xp)
        xs = self.vconv(xs)
        return self.fconv(torch.cat((xp, xs), dim=1))


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Functional form of stochastic depth."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """Module wrapper for stochastic depth."""
    def __init__(self, drop_prob: float | None = None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob or 0.0, self.training)


class PatchEmbed(nn.Module):
    """Split seismic records into patches and map them into Transformer embeddings."""
    def __init__(self, nt: int, nr: int, patch_size: Tuple[int, int] = (16, 16), embed_dim: int = 768, norm_layer=None):
        super().__init__()
        self.num_patches = (nt // patch_size[0]) * (nr // patch_size[1])
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)


class Attention(nn.Module):
    """Standard multi-head self-attention."""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale=None,
        attn_drop_ratio: float = 0.0,
        proj_drop_ratio: float = 0.0,
    ):
        super().__init__()
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(bsz, n_tokens, 3, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(bsz, n_tokens, channels)
        x = self.proj(x)
        return self.proj_drop(x)


class Mlp(nn.Module):
    """Feed-forward network used inside the Transformer block."""
    def __init__(self, in_features: int, hidden_features: int | None = None, out_features: int | None = None, act_layer=nn.GELU, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class Block(nn.Module):
    """Standard Transformer block: attention + MLP."""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale=None,
        drop_ratio: float = 0.0,
        attn_drop_ratio: float = 0.0,
        drop_path_ratio: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_ratio=attn_drop_ratio,
            proj_drop_ratio=drop_ratio,
        )
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ClampLayer(nn.Module):
    """Clamp the predicted velocity to an upper bound."""
    def __init__(self, max_value: float):
        super().__init__()
        self.max_value = nn.Parameter(torch.tensor(max_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, max=self.max_value)


class DecoderVP(nn.Module):
    """Upsample Transformer features back to the velocity-model grid."""
    def __init__(self, batch_size: int, initial_shape: Tuple[int, int], final_shape: Tuple[int, int], n_blocks: int, final_out_channels: int = 1):
        super().__init__()
        self.initial_shape = tuple(int(v) for v in initial_shape)
        self.batch_size = batch_size
        out_channels = sorted([8 * (2 ** i) for i in range(n_blocks)], reverse=True)
        layers = [Eblock(1, out_channels[0], stride=1, operation="up")]
        for layer_idx in range(n_blocks - 1):
            finalize = final_shape if layer_idx == n_blocks - 2 else None
            layers.append(Eblock(out_channels[layer_idx], out_channels[layer_idx + 1], stride=1, operation="up", final_shape=finalize))
        self.conv_layers = nn.Sequential(*layers)
        self.final = nn.Sequential(
            nn.Conv2d(out_channels[-1], final_out_channels, kernel_size=3, padding=1, stride=1, bias=True),
            ClampLayer(5000.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(self.batch_size, 1, self.initial_shape[0], self.initial_shape[1])
        return self.final(self.conv_layers(x))


class LowRankTensorGenerator(nn.Module):
    """Build low-rank-like features by reducing and expanding channels."""
    def __init__(self, input_channels: int, rank_factor: float = 0.5):
        super().__init__()
        reduced_channels = int(input_channels * rank_factor)
        self.reduce = nn.Conv2d(input_channels, reduced_channels, kernel_size=1)
        self.expand = nn.Conv2d(reduced_channels, input_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expand(self.reduce(x))


class ChannelFusion(nn.Module):
    """Single-input channel-attention fusion module."""
    def __init__(self, input_channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(input_channels, input_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(input_channels // 2, input_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attention(x)


class SingleInputFusionNet(nn.Module):
    """Preprocessing network with low-rank generation followed by channel fusion."""
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.low_rank_gen = LowRankTensorGenerator(input_channels)
        self.fusion = ChannelFusion(input_channels)
        self.final_conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.low_rank_gen(x)
        x = self.fusion(x)
        return self.final_conv(x)


class TransformerDecoder(nn.Module):
    """Main network.

    It takes seismic records as input and predicts a velocity model.
    """
    def __init__(
        self,
        batch_size: int,
        in_channels: int,
        nt: int,
        nr: int,
        patch_size: Tuple[int, int] = (16, 16),
        embed_dim: int = 768,
        transddepth: int = 1,
        n_blocks_decoder: int = 4,
        final_size_encoder: int = 98,
        initial_shape_decoder: Tuple[int, int] = (14, 28),
        final_spatial_shape: Tuple[int, int] = (116, 227),
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale=None,
        drop_ratio: float = 0.0,
        attn_drop_ratio: float = 0.0,
        drop_path_ratio: float = 0.0,
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        # 2D grid size after patch embedding.
        self.h_v = nt // patch_size[0]
        self.w_v = nr // patch_size[1]
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        # Compress input channels before patch embedding.
        self.fusion_net = SingleInputFusionNet(input_channels=in_channels, output_channels=1)
        self.patch_embed = embed_layer(nt, nr, patch_size=patch_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, transddepth)]
        # Transformer backbone.
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_ratio=drop_ratio,
                    attn_drop_ratio=attn_drop_ratio,
                    drop_path_ratio=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(n_blocks_decoder)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.fc_in_features = batch_size * embed_dim * self.h_v * self.w_v
        self.final = nn.Sequential(nn.Linear(in_features=self.fc_in_features, out_features=final_size_encoder))
        # Restore the spatial resolution of the velocity model.
        self.decoder_vp = DecoderVP(
            batch_size=batch_size,
            initial_shape=tuple(int(v) for v in initial_shape_decoder),
            final_shape=tuple(int(v) for v in final_spatial_shape),
            n_blocks=n_blocks_decoder,
            final_out_channels=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: fusion -> patches -> Transformer -> linear map -> decoder."""
        x = self.fusion_net(x)
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        bsz, channels = x.shape[0], x.shape[2]
        x = x.reshape(bsz, channels, int(self.h_v), int(self.w_v))
        x = x.reshape(-1)
        x = self.final(x)
        return self.decoder_vp(x)


class SiameseNetwork(nn.Module):
    """Siamese network used in stage two to compare synthetic and observed records."""
    def __init__(self, D1: int):
        super().__init__()
        self.D1 = D1
        self.cnn1 = nn.Conv2d(1, self.D1, kernel_size=3, stride=1, padding=1)
        self.a1 = nn.LeakyReLU(0.5)
        self.cnn2 = nn.Conv2d(self.D1, 2 * self.D1, kernel_size=3, stride=1, padding=1)
        self.a2 = nn.LeakyReLU(0.5)
        self.cnn3 = nn.Conv2d(2 * self.D1, 2 * self.D1, kernel_size=3, stride=1, padding=1)
        self.a3 = nn.LeakyReLU(0.5)
        self.cnn4 = nn.Conv2d(2 * self.D1, 4 * self.D1, kernel_size=3, stride=1, padding=1)
        self.a4 = nn.LeakyReLU(0.5)
        self.cnn5 = nn.Conv2d(4 * self.D1, 4 * self.D1, kernel_size=3, stride=1, padding=1)
        self.a5 = nn.LeakyReLU(0.5)
        self.cnn6 = nn.Conv2d(4 * self.D1, 2 * self.D1, kernel_size=3, stride=1, padding=1)
        self.a6 = nn.LeakyReLU(0.5)
        self.cnn7 = nn.Conv2d(2 * self.D1, self.D1, kernel_size=3, stride=1, padding=1)
        self.a7 = nn.LeakyReLU(0.5)
        self.cnn8 = nn.Conv2d(self.D1, 1, kernel_size=3, stride=1, padding=1)
        self.cnnXX1 = nn.Conv2d(1, self.D1, kernel_size=3, stride=1, padding=1)
        self.axx1 = nn.LeakyReLU(0.5)
        self.cnnXX2 = nn.Conv2d(1, 2 * self.D1, kernel_size=3, stride=1, padding=1)
        self.axx2 = nn.LeakyReLU(0.5)
        self.cnnXX3 = nn.Conv2d(1, 2 * self.D1, kernel_size=3, stride=1, padding=1)
        self.axx3 = nn.LeakyReLU(0.5)
        self.cnnXX4 = nn.Conv2d(1, 4 * self.D1, kernel_size=3, stride=1, padding=1)
        self.axx4 = nn.LeakyReLU(0.5)
        self.cnnXX5 = nn.Conv2d(1, 4 * self.D1, kernel_size=3, stride=1, padding=1)
        self.axx5 = nn.LeakyReLU(0.5)
        self.cnnXX6 = nn.Conv2d(1, 2 * self.D1, kernel_size=3, stride=1, padding=1)
        self.axx6 = nn.LeakyReLU(0.5)
        self.cnnXX7 = nn.Conv2d(1, self.D1, kernel_size=3, stride=1, padding=1)
        self.axx7 = nn.LeakyReLU(0.5)
        self.cnnXX = nn.Conv2d(1, self.D1, kernel_size=3, stride=1, padding=1)

    def forward_once(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xx1 = self.axx1(self.cnnXX1(x))
        xx2 = self.axx2(self.cnnXX2(x))
        xx3 = self.axx3(self.cnnXX3(x))
        xx4 = self.axx4(self.cnnXX4(x))
        xx5 = self.axx5(self.cnnXX5(x))
        xx6 = self.axx6(self.cnnXX6(x))
        xx7 = self.axx7(self.cnnXX7(x))
        xx = self.cnnXX(x)
        output1 = self.a1(self.cnn1(x)) + xx1
        output2 = self.a2(self.cnn2(output1)) + xx2
        output3 = self.a3(self.cnn3(output2)) + xx3
        output4 = self.a4(self.cnn4(output3)) + xx4
        output5 = self.a5(self.cnn5(output4)) + xx5
        output6 = self.a6(self.cnn6(output5)) + xx6
        output7 = self.a7(self.cnn7(output6)) + xx7
        output8 = self.cnn8(output7) + xx + x
        return output4, output8

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        output1, o1 = self.forward_once(input1)
        output2, o2 = self.forward_once(input2)
        return output1, output2, o1, o2


Decoder_vp = DecoderVP
Transfomerdecoder = TransformerDecoder
