import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import List, Tuple

# 定义 DropPath 模块
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.size(0),) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)

# 定义 CMNeXt1 模块的组件
class Attention(nn.Module):
    def __init__(self, dim, head, sr_ratio):
        super().__init__()
        self.head = head
        self.sr_ratio = sr_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
        else:
            x_ = x

        kv = self.kv(x_).reshape(B, -1, 2, self.head, C // self.head)
        k, v = kv.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        x = self.proj(x)
        return x

class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, N, C = x.shape
        x = x.transpose(1,2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1,2)
        return x

class FeedForward(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x: Tensor, H, W) -> Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dwconv(x, H, W)
        x = self.fc2(x)
        return x

class LayerNormParallel(nn.Module):
    def __init__(self, num_features, num_modals=2):
        super().__init__()
        self.ln = nn.ModuleList([nn.LayerNorm(num_features) for _ in range(num_modals)])

    def forward(self, x_parallel):
        return [ln(xi) for ln, xi in zip(self.ln, x_parallel)]

class PatchEmbedParallel(nn.Module):
    def __init__(self, c1: List[int], c2: int, patch_size: int, stride: int, padding: int, num_modals: int):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Conv2d(c1[i], c2, kernel_size=patch_size, stride=stride, padding=padding)
            for i in range(num_modals)
        ])
        self.norm = LayerNormParallel(c2, num_modals)

    def forward(self, x: List[Tensor]) -> Tuple[List[Tensor], int, int]:
        x = [proj(xi) for proj, xi in zip(self.proj, x)]
        _, _, H, W = x[0].shape
        x = [xi.flatten(2).transpose(1, 2) for xi in x]
        x = self.norm(x)
        return x, H, W

class Block(nn.Module):
    def __init__(self, dim, head, sr_ratio=1, dpr=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, int(dim * 4))

    def forward(self, x: Tensor, H, W) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

cmnext_settings = {
    'B2': [[64, 128, 320, 512], [3, 4, 6, 3]],
    'B4': [[64, 128, 320, 512], [3, 8, 27, 3]],
    'B5': [[64, 128, 320, 512], [3, 6, 40, 3]]
}

class CMNeXt1(nn.Module):
    def __init__(self, model_name: str = 'B2', num_modals: int = 2):
        super().__init__()
        assert model_name in cmnext_settings.keys(), f"Model name should be in {list(cmnext_settings.keys())}"
        embed_dims, depths = cmnext_settings[model_name]
        self.num_modals = num_modals
        drop_path_rate = 0.1

        # Patch embedding for multiple stages
        self.patch_embed1 = PatchEmbedParallel([3, 1], embed_dims[0], 7, 4, 3, self.num_modals)
        self.patch_embed2 = PatchEmbedParallel([embed_dims[0]] * self.num_modals, embed_dims[1], 3, 2, 1, self.num_modals)
        self.patch_embed3 = PatchEmbedParallel([embed_dims[1]] * self.num_modals, embed_dims[2], 3, 2, 1, self.num_modals)
        self.patch_embed4 = PatchEmbedParallel([embed_dims[2]] * self.num_modals, embed_dims[3], 3, 2, 1, self.num_modals)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Defining blocks for each stage
        cur = 0
        self.block1 = nn.ModuleList([Block(embed_dims[0], 1, 8, dpr[cur + i]) for i in range(depths[0])])
        self.norm1 = LayerNormParallel(embed_dims[0], self.num_modals)
        cur += depths[0]

        self.block2 = nn.ModuleList([Block(embed_dims[1], 2, 4, dpr[cur + i]) for i in range(depths[1])])
        self.norm2 = LayerNormParallel(embed_dims[1], self.num_modals)
        cur += depths[1]

        self.block3 = nn.ModuleList([Block(embed_dims[2], 5, 2, dpr[cur + i]) for i in range(depths[2])])
        self.norm3 = LayerNormParallel(embed_dims[2], self.num_modals)
        cur += depths[2]

        self.block4 = nn.ModuleList([Block(embed_dims[3], 8, 1, dpr[cur + i]) for i in range(depths[3])])
        self.norm4 = LayerNormParallel(embed_dims[3], self.num_modals)
        cur += depths[3]

    def forward(self, x: List[Tensor]) -> List[List[Tensor]]:
        B = x[0].shape[0]
        outs = []

        # Stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = [blk(xi, H, W) for xi in x]
        x1 = [norm(xi).reshape(B, H, W, -1).permute(0, 3, 1, 2) for norm, xi in zip(self.norm1.ln, x)]
        outs.append(x1)

        # Stage 2
        x, H, W = self.patch_embed2(x1)
        for blk in self.block2:
            x = [blk(xi, H, W) for xi in x]
        x2 = [norm(xi).reshape(B, H, W, -1).permute(0, 3, 1, 2) for norm, xi in zip(self.norm2.ln, x)]
        outs.append(x2)

        # Stage 3
        x, H, W = self.patch_embed3(x2)
        for blk in self.block3:
            x = [blk(xi, H, W) for xi in x]
        x3 = [norm(xi).reshape(B, H, W, -1).permute(0, 3, 1, 2) for norm, xi in zip(self.norm3.ln, x)]
        outs.append(x3)

        # Stage 4
        x, H, W = self.patch_embed4(x3)
        for blk in self.block4:
            x = [blk(xi, H, W) for xi in x]
        x4 = [norm(xi).reshape(B, H, W, -1).permute(0, 3, 1, 2) for norm, xi in zip(self.norm4.ln, x)]
        outs.append(x4)

        return outs

# 定义 SegFormerHead 模块
class MLP_SegFormer(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class ConvModule(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activate(self.bn(self.conv(x)))

class SegFormerHead(nn.Module):
    def __init__(self, dims: List[int], embed_dim: int = 256, num_classes: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.linear_c = nn.ModuleList([MLP_SegFormer(dim, embed_dim) for dim in dims])

        self.linear_fuse = ConvModule(embed_dim * len(dims), embed_dim)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features: List[List[Tensor]]) -> Tensor:
        # 融合不同模态的特征
        fused_features = []
        for stage_features in features:
            # 将不同模态的特征相加
            fused = sum(stage_features)
            fused_features.append(fused)

        # 处理融合后的特征
        B, C, H, W = fused_features[0].shape
        outs = []
        for i, feature in enumerate(fused_features):
            x = self.linear_c[i](feature)
            x = x.permute(0, 2, 1).reshape(B, -1, feature.shape[2], feature.shape[3])
            if i > 0:
                x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
            outs.append(x)
        x = torch.cat(outs[::-1], dim=1)
        x = self.linear_fuse(x)
        x = self.dropout(x)
        x = self.linear_pred(x)
        return x

# 将 CMNeXt1 和 SegFormerHead 整合到 CMNeXt 模型中
class CMNEXT(nn.Module):
    def __init__(self, model_name: str = 'B2', num_classes: int = 1):
        super().__init__()
        self.backbone = CMNeXt1(model_name, num_modals=2)
        dims = cmnext_settings[model_name][0]  # [64, 128, 320, 512]
        self.decode_head = SegFormerHead(dims, embed_dim=256, num_classes=num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        # 按照您的要求，将输入拆分为 x 和 add
        x = inputs[:, :3, :, :]  # 图像 [batch_size, 3, H, W]
        add = inputs[:, 3:, :, :]  # GPS 或 LiDAR 数据 [batch_size, 1, H, W]

        # 将两个模态的输入组成列表
        x_modalities = [x, add]

        # 通过主干网络提取特征
        features = self.backbone(x_modalities)

        # 通过解码头获得输出
        out = self.decode_head(features)

        # 将输出调整到与输入相同的尺寸
        out = F.interpolate(out, size=inputs.shape[2:], mode='bilinear', align_corners=False)
        return torch.sigmoid(out)

# 测试代码
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 输入形状为 [4, 4, 512, 512]
    inputs = torch.randn(4, 4, 512, 512).to(device)
    model = CMNEXT('B2', num_classes=1).to(device)
    out = model(inputs)
    print("Output shape:", out.shape)
    # 期望输出形状：[4, 1, 512, 512]
