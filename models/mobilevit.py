# plant_disease_classifier/models/mobilevit.py
import torch
import torch.nn as nn
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
        nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

# ----------------------
# Transformer Blocks
# ----------------------
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        # x: [batch, tokens, dim]
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
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
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
        # x: [batch, tokens, dim]
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads),
            qkv
        )
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# ----------------------
# MobileNetV2 Block
# ----------------------
class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        hidden_dim = inp * expansion
        self.use_res_connect = (stride == 1 and inp == oup)

        if expansion == 1:
            self.conv = nn.Sequential(
                # depthwise
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pointwise-linear
                nn.Conv2d(hidden_dim, oup, kernel_size=1, bias=False),
                nn.BatchNorm2d(oup)
            )
        else:
            self.conv = nn.Sequential(
                # pointwise
                nn.Conv2d(inp, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # depthwise
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pointwise-linear
                nn.Conv2d(hidden_dim, oup, kernel_size=1, bias=False),
                nn.BatchNorm2d(oup)
            )

    def forward(self, x):
        return x + self.conv(x) if self.use_res_connect else self.conv(x)

# ----------------------
# MobileViT Block
# ----------------------
class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size
        self.conv_local = conv_nxn_bn(channel, channel, kernel_size)
        self.conv_project = conv_1x1_bn(channel, dim)
        self.transformer = Transformer(dim, depth, heads=4, dim_head=dim//4, mlp_dim=mlp_dim, dropout=dropout)
        self.conv_expand = conv_1x1_bn(dim, channel)
        self.conv_fuse = conv_nxn_bn(2*channel, channel, kernel_size)

    def forward(self, x):
        residual = x
        # local processing
        x = self.conv_local(x)
        x = self.conv_project(x)

        # unfold into patches
        b, c, h, w = x.shape
        ph, pw = self.ph, self.pw
        x = rearrange(x, 'b c (h ph) (w pw) -> b (ph pw) (h w) c', ph=ph, pw=pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) c -> b c (h ph) (w pw)', h=h//ph, w=w//pw, ph=ph, pw=pw)

        # fuse
        x = self.conv_expand(x)
        x = torch.cat((x, residual), dim=1)
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
        patch_size: tuple[int, int] = (2, 2)
    ) -> None:
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0, "Image dimensions must be divisible by patch size"

        # initial conv
        self.conv1 = conv_nxn_bn(3, channels[0], stride=2)

        # MV2 segments
        self.mv2 = nn.ModuleList([
            MV2Block(channels[0], channels[1], stride=1, expansion=expansion),
            MV2Block(channels[1], channels[2], stride=2, expansion=expansion),
            MV2Block(channels[2], channels[3], stride=1, expansion=expansion),
            MV2Block(channels[3], channels[3], stride=1, expansion=expansion),
            MV2Block(channels[3], channels[4], stride=2, expansion=expansion),
            MV2Block(channels[4], channels[5], stride=1, expansion=expansion),
            MV2Block(channels[5], channels[6], stride=2, expansion=expansion)
        ])

        # MobileViT blocks
        self.mvit = nn.ModuleList([
            MobileViTBlock(dims[0], depth=2, channel=channels[2], kernel_size=kernel_size, patch_size=patch_size, mlp_dim=dims[0]*2),
            MobileViTBlock(dims[1], depth=4, channel=channels[4], kernel_size=kernel_size, patch_size=patch_size, mlp_dim=dims[1]*4),
            MobileViTBlock(dims[2], depth=3, channel=channels[6], kernel_size=kernel_size, patch_size=patch_size, mlp_dim=dims[2]*4)
        ])

        # fusion
        self.conv2 = conv_1x1_bn(channels[6], channels[7])

        # classifier head
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(channels[7], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 3, H, W]
        x = self.conv1(x)
        for layer in self.mv2:
            x = layer(x)

        for block in self.mvit:
            x = block(x)

        x = self.conv2(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)

# ----------------------
# Classifier Wrapper
# ----------------------
class MobileViTClassifier(nn.Module):
    def __init__(self, image_size: tuple[int,int], num_classes: int, expansion: int = 4) -> None:
        super().__init__()
        # define dims and channels for small model
        dims = [64, 80, 96]
        channels = [16, 16, 24, 24, 48, 48, 64, 320]
        self.model = MobileViT(image_size=image_size, dims=dims, channels=channels, num_classes=num_classes, expansion=expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)