# models/mobilevit.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ----------------------
# Helper Convolutions
# ----------------------
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def conv_nxn_bn(inp, oup, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=stride,
                  padding=kernel_size // 2, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

# ----------------------
# Transformer Blocks
# ----------------------
class PreNorm(nn.Module):
    """LayerNorm + Submodule (expects input [B, N, D])."""
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    """Multi-Head Self-Attention for tokens [B, N, D]."""
    def __init__(self, dim, heads=2, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)                 # 3 x [B, N, H*Dh]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)                           # [B, H, N, Dh]
        out = rearrange(out, 'b h n d -> b n (h d)')          # [B, N, D]
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]) for _ in range(depth)
        ])
    def forward(self, x):
        for attn, ff in self.layers:
            x = x + attn(x)
            x = x + ff(x)
        return x

# ----------------------
# MobileNetV2 Block
# ----------------------
class MV2Block(nn.Module):
    """
    Inverted residual block. If stride==1 and inp==oup, uses residual.
    """
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        hidden_dim = inp * expansion
        self.use_res_connect = (stride == 1 and inp == oup)

        if expansion == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=3, stride=stride, padding=1,
                          groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.SiLU(),
                nn.Conv2d(inp, oup, kernel_size=1, bias=False),
                nn.BatchNorm2d(oup)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, oup, kernel_size=1, bias=False),
                nn.BatchNorm2d(oup)
            )

    def forward(self, x):
        out = self.conv(x)
        return x + out if self.use_res_connect else out

# ----------------------
# MobileViT Block (patch-tokenized to cut memory)
# ----------------------
class MobileViTBlock(nn.Module):
    """
    Local CNN -> 1x1 project to D -> **patch-token** Transformer -> 1x1 expand -> fuse with residual.
    Patch tokens reduce sequence length by (ph*pw) vs pixel tokens, avoiding OOM on small GPUs.
    """
    def __init__(self, dim, depth, channel, kernel_size, mlp_dim, dropout=0.,
                 patch_size=(4, 4), heads=2, dim_head=32, token_mode: str = "patch"):
        super().__init__()
        assert token_mode in {"patch", "pixel"}
        self.token_mode = token_mode
        self.ph, self.pw = patch_size

        self.conv_local   = conv_nxn_bn(channel, channel, kernel_size)
        self.conv_project = conv_1x1_bn(channel, dim)

        # Transformer
        self.transformer = Transformer(dim, depth, heads=heads, dim_head=dim_head,
                                       mlp_dim=mlp_dim, dropout=dropout)

        # For patch mode: linear in/out of tokens
        self.to_token = nn.Linear(dim * self.ph * self.pw, dim)
        self.to_patch = nn.Linear(dim, dim * self.ph * self.pw)

        self.conv_expand = conv_1x1_bn(dim, channel)
        self.conv_fuse   = conv_nxn_bn(channel * 2, channel, kernel_size)

    def forward(self, x):
        residual = x
        x = self.conv_local(x)                 # [B, C, H, W]
        x = self.conv_project(x)               # [B, D, H, W]
        b, d, h, w = x.shape

        if self.token_mode == "pixel":
            # (B, D, H, W) -> (B, N, D)
            tokens = rearrange(x, 'b d h w -> b (h w) d')
            tokens = self.transformer(tokens)
            x = rearrange(tokens, 'b (h w) d -> b d h w', h=h, w=w)
        else:
            # patch-tokenized path
            ph, pw = self.ph, self.pw
            pad_h = (ph - h % ph) % ph
            pad_w = (pw - w % pw) % pw
            if pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h))
                residual = F.pad(residual, (0, pad_w, 0, pad_h))
                h, w = x.shape[-2:]
            hb, wb = h // ph, w // pw

            # (B, D, H, W) -> (B, N, ph*pw*D) -> (B, N, D)
            tokens = rearrange(x, 'b d (h ph) (w pw) -> b (h w) (ph pw d)', ph=ph, pw=pw)
            tokens = self.to_token(tokens)
            tokens = self.transformer(tokens)
            tokens = self.to_patch(tokens)
            # back to map
            x = rearrange(tokens, 'b (h w) (ph pw d) -> b d (h ph) (w pw)', h=hb, w=wb, ph=ph, pw=pw)

            if pad_h or pad_w:
                x = x[:, :, : (h - pad_h), : (w - pad_w)]
                residual = residual[:, :, : (h - pad_h), : (w - pad_w)]

        x = self.conv_expand(x)                # [B, C, H, W]
        x = torch.cat((x, residual), dim=1)    # fuse local+global
        return self.conv_fuse(x)

# ----------------------
# MobileViT Architecture
# ----------------------
class MobileViT(nn.Module):
    def __init__(
        self,
        image_size: tuple[int, int],
        dims: list[int],
        channels: list[int],
        num_classes: int,
        expansion: int = 4,
        kernel_size: int = 3,
        patch_size: tuple[int, int] = (4, 4),   # <-- memory-safe default
        heads: int = 2,
        dim_head: int = 32,
        dropout: float = 0.0,
        token_mode: str = "patch",              # <-- patch tokens by default
    ) -> None:
        super().__init__()

        # Stem
        self.conv1 = conv_nxn_bn(3, channels[0], stride=2)

        # MV2 stages
        self.mv2 = nn.ModuleList([
            MV2Block(channels[0], channels[1], stride=1, expansion=expansion),
            MV2Block(channels[1], channels[2], stride=2, expansion=expansion),
            MV2Block(channels[2], channels[3], stride=1, expansion=expansion),
            MV2Block(channels[3], channels[3], stride=1, expansion=expansion),
            MV2Block(channels[3], channels[4], stride=2, expansion=expansion),
            MV2Block(channels[4], channels[5], stride=1, expansion=expansion),
            MV2Block(channels[5], channels[6], stride=2, expansion=expansion),
        ])

        # MobileViT blocks at three scales
        self.mvit = nn.ModuleList([
            MobileViTBlock(dims[0], depth=2, channel=channels[2], kernel_size=kernel_size,
                           mlp_dim=dims[0] * 2, dropout=dropout, patch_size=patch_size,
                           heads=heads, dim_head=dim_head, token_mode=token_mode),
            MobileViTBlock(dims[1], depth=4, channel=channels[4], kernel_size=kernel_size,
                           mlp_dim=dims[1] * 4, dropout=dropout, patch_size=patch_size,
                           heads=heads, dim_head=dim_head, token_mode=token_mode),
            MobileViTBlock(dims[2], depth=3, channel=channels[6], kernel_size=kernel_size,
                           mlp_dim=dims[2] * 4, dropout=dropout, patch_size=patch_size,
                           heads=heads, dim_head=dim_head, token_mode=token_mode),
        ])

        # Fusion + Head
        self.conv2 = conv_1x1_bn(channels[6], channels[7])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(channels[7], num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.conv1(x)

        # Stages with interleaved MobileViT blocks
        x = self.mv2[0](x)                                 # 16 -> 16 (s=1)

        x = self.mv2[1](x)                                 # 16 -> 24 (s=2)
        x = self.mv2[2](x)                                 # 24 -> 24 (s=1)
        x = self.mvit[0](x)                                # ---- MobileViT @ 24-ch ----

        x = self.mv2[3](x)                                 # 24 -> 24 (s=1)
        x = self.mv2[4](x)                                 # 24 -> 48 (s=2)
        x = self.mv2[5](x)                                 # 48 -> 48 (s=1)
        x = self.mvit[1](x)                                # ---- MobileViT @ 48-ch ----

        x = self.mv2[6](x)                                 # 48 -> 64 (s=2)
        x = self.mvit[2](x)                                # ---- MobileViT @ 64-ch ----

        x = self.conv2(x)                                  # 64 -> 320
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)

# ----------------------
# Classifier Wrapper
# ----------------------
class MobileViTClassifier(nn.Module):
    """
    Small MobileViT configuration (≈ MobileViT-XXS-like) for 224x224 inputs.
    Uses patch-token attention by default to reduce memory on small GPUs.
    """
    def __init__(self, image_size: tuple[int,int], num_classes: int, expansion: int = 4,
                 dropout: float = 0.0, patch_size=(4,4), heads=2, dim_head=32,
                 token_mode: str = "patch") -> None:
        super().__init__()
        dims = [64, 80, 96]
        channels = [16, 16, 24, 24, 48, 48, 64, 320]
        self.model = MobileViT(
            image_size=image_size,
            dims=dims,
            channels=channels,
            num_classes=num_classes,
            expansion=expansion,
            kernel_size=3,
            patch_size=patch_size,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            token_mode=token_mode
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
