import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights
from .basic_blocks import *

class SPPLayer(torch.nn.Module):
    def __init__(self, block_size=[1,2,4], pool_type='max_pool'):
        super(SPPLayer, self).__init__()
        self.block_size = block_size
        self.pool_type = pool_type
        self.spp = self.make_spp(out_pool_size=self.block_size, pool_type=self.pool_type)

    def make_spp(self, out_pool_size, pool_type='maxpool'):
        func=[]
        for i in range(len(out_pool_size)):
            if pool_type == 'max_pool':
                func.append(nn.AdaptiveMaxPool2d(output_size=(out_pool_size[i],out_pool_size[i])))
            if pool_type == 'avg_pool':
                func.append(nn.AdaptiveAvgPool2d(output_size=(out_pool_size[i],out_pool_size[i])))
        return func

    def forward(self, x):
        num = x.size(0)
        for i in range(len(self.block_size)):
            tensor = self.spp[i](x).view(num, -1)
            if (i == 0):
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
        return x_flatten

class DEM(torch.nn.Module):  # Dual Enhancement Module
    def __init__(self, channel, block_size=[1,2,4]):
        super(DEM, self).__init__()

        self.rgb_local_message = self.local_message_prepare(channel, 1, 1, 0)
        self.add_local_message = self.local_message_prepare(channel, 1, 1, 0)

        self.rgb_spp = SPPLayer(block_size=block_size)
        self.add_spp = SPPLayer(block_size=block_size)
        self.rgb_global_message = self.global_message_prepare(block_size, channel)
        self.add_global_message = self.global_message_prepare(block_size, channel)

        self.rgb_local_gate  = self.gate_build(channel*2, channel, 1, 1, 0)
        self.rgb_global_gate = self.gate_build(channel*2, channel, 1, 1, 0)

        self.add_local_gate  = self.gate_build(channel*2, channel, 1, 1, 0)
        self.add_global_gate = self.gate_build(channel*2, channel, 1, 1, 0)

    def local_message_prepare(self, dim, kernel_size=3, stride=1, padding=1, bias=True):
        return nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(dim)
        )

    def global_message_prepare(self, block_size, dim):
        num_block = sum([i * i for i in block_size])
        return nn.Sequential(
            nn.Linear(num_block * dim, dim),
            nn.ReLU()
        )

    def gate_build(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding),
            nn.Sigmoid()
        )

    def forward(self, rgb_info, add_info):
        rgb_local_info = self.rgb_local_message(rgb_info)
        add_local_info = self.add_local_message(add_info)

        rgb_spp_output = self.rgb_spp(rgb_local_info)
        add_spp_output = self.add_spp(add_local_info)

        rgb_global_info = self.rgb_global_message(rgb_spp_output).unsqueeze(-1).unsqueeze(-1).expand_as(rgb_local_info)
        add_global_info = self.add_global_message(add_spp_output).unsqueeze(-1).unsqueeze(-1).expand_as(add_local_info)

        rgb_gate_input = torch.cat((add_local_info, add_global_info), dim=1)
        add_gate_input = torch.cat((rgb_local_info, rgb_global_info), dim=1)

        rgb_info = rgb_info + add_local_info * self.add_local_gate(rgb_gate_input) + add_global_info * self.add_global_gate(rgb_gate_input)
        add_info = add_info + rgb_local_info * self.rgb_local_gate(add_gate_input) + rgb_global_info * self.rgb_global_gate(add_gate_input)

        return rgb_info, add_info

class CMMPNet(nn.Module):
    def __init__(self, block_size='1,2,4', num_classes=2):
        super(CMMPNet, self).__init__()
        filters = [64, 128, 256, 512]
        self.net_name = "CMMPNet"
        self.block_size = [int(s) for s in block_size.split(',')]
        self.num_classes = num_classes

        # 定义激活函数
        self.nonlinearity = F.relu

        # 图像分支
        weights = ResNet34_Weights.DEFAULT
        resnet = resnet34(weights=weights)
        self.firstconv1 = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = DBlock(filters[3])

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(
            filters[0], filters[0] // 2, 4, 2, 1)
        self.finalrelu1 = self.nonlinearity
        self.finalconv2 = nn.Conv2d(
            filters[0] // 2, filters[0] // 2, 3, padding=1)
        self.finalrelu2 = self.nonlinearity

        # 辅助信息分支，例如 GPS 图或激光雷达图
        resnet1 = resnet34(weights=weights)
        self.firstconv1_add = nn.Conv2d(
            1, filters[0], kernel_size=7, stride=2, padding=3)
        self.firstbn_add = resnet1.bn1
        self.firstrelu_add = resnet1.relu
        self.firstmaxpool_add = resnet1.maxpool

        self.encoder1_add = resnet1.layer1
        self.encoder2_add = resnet1.layer2
        self.encoder3_add = resnet1.layer3
        self.encoder4_add = resnet1.layer4

        self.dblock_add = DBlock(filters[3])

        self.decoder4_add = DecoderBlock(filters[3], filters[2])
        self.decoder3_add = DecoderBlock(filters[2], filters[1])
        self.decoder2_add = DecoderBlock(filters[1], filters[0])
        self.decoder1_add = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1_add = nn.ConvTranspose2d(
            filters[0], filters[0] // 2, 4, 2, 1)
        self.finalrelu1_add = self.nonlinearity
        self.finalconv2_add = nn.Conv2d(
            filters[0] // 2, filters[0] // 2, 3, padding=1)
        self.finalrelu2_add = self.nonlinearity

        # DEM 模块
        self.dem_e1 = DEM(filters[0], self.block_size)
        self.dem_e2 = DEM(filters[1], self.block_size)
        self.dem_e3 = DEM(filters[2], self.block_size)
        self.dem_e4 = DEM(filters[3], self.block_size)

        self.dem_d4 = DEM(filters[2], self.block_size)
        self.dem_d3 = DEM(filters[1], self.block_size)
        self.dem_d2 = DEM(filters[0], self.block_size)
        self.dem_d1 = DEM(filters[0], self.block_size)

        # 定义用于计算证据和概率的卷积层
        self.evidence_layer_x = nn.Conv2d(
            filters[0] // 2, self.num_classes, kernel_size=1)
        self.evidence_layer_add = nn.Conv2d(
            filters[0] // 2, self.num_classes, kernel_size=1)
        self.b12_layer_x = nn.Conv2d(
            filters[0] // 2, self.num_classes, kernel_size=1)
        self.b12_layer_add = nn.Conv2d(
            filters[0] // 2, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        x = inputs[:, :3, :, :]    # 图像
        add = inputs[:, 3:, :, :]  # GPS 图或激光雷达图

        x = self.firstconv1(x)
        add = self.firstconv1_add(add)
        x = self.firstmaxpool(self.firstrelu(self.firstbn(x)))
        add = self.firstmaxpool_add(self.firstrelu_add(self.firstbn_add(add)))

        x_e1 = self.encoder1(x)
        add_e1 = self.encoder1_add(add)
        x_e1, add_e1 = self.dem_e1(x_e1, add_e1)

        x_e2 = self.encoder2(x_e1)
        add_e2 = self.encoder2_add(add_e1)
        x_e2, add_e2 = self.dem_e2(x_e2, add_e2)

        x_e3 = self.encoder3(x_e2)
        add_e3 = self.encoder3_add(add_e2)
        x_e3, add_e3 = self.dem_e3(x_e3, add_e3)

        x_e4 = self.encoder4(x_e3)
        add_e4 = self.encoder4_add(add_e3)
        x_e4, add_e4 = self.dem_e4(x_e4, add_e4)

        # 中心部分
        x_c = self.dblock(x_e4)
        add_c = self.dblock_add(add_e4)

        # 解码器
        x_d4 = self.decoder4(x_c) + x_e3
        add_d4 = self.decoder4_add(add_c) + add_e3
        x_d4, add_d4 = self.dem_d4(x_d4, add_d4)

        x_d3 = self.decoder3(x_d4) + x_e2
        add_d3 = self.decoder3_add(add_d4) + add_e2
        x_d3, add_d3 = self.dem_d3(x_d3, add_d3)

        x_d2 = self.decoder2(x_d3) + x_e1
        add_d2 = self.decoder2_add(add_d3) + add_e1
        x_d2, add_d2 = self.dem_d2(x_d2, add_d2)

        x_d1 = self.decoder1(x_d2)
        add_d1 = self.decoder1_add(add_d2)
        x_d1, add_d1 = self.dem_d1(x_d1, add_d1)

        x_out = self.finalrelu1(self.finaldeconv1(x_d1))
        add_out = self.finalrelu1_add(self.finaldeconv1_add(add_d1))
        x_out = self.finalrelu2(self.finalconv2(x_out))
        add_out = self.finalrelu2_add(self.finalconv2_add(add_out))

        # 计算每个模态的证据和概率
        e1 = F.softplus(self.evidence_layer_x(x_out))  # (B, num_classes, H, W)
        e2 = F.softplus(self.evidence_layer_add(add_out))  # (B, num_classes, H, W)

        b12_1 = F.softmax(self.b12_layer_x(x_out), dim=1)  # (B, num_classes, H, W)
        b12_2 = F.softmax(self.b12_layer_add(add_out), dim=1)  # (B, num_classes, H, W)

        # 计算第一个模态的 Dirichlet 分布参数
        epsilon = 1e-8  # 防止除零
        alpha11 = e1 + 1  # (B, num_classes, H, W)
        S1 = torch.sum(alpha11, dim=1, keepdim=True)  # (B, 1, H, W)
        S1 = S1 + epsilon
        b11 = e1 / S1  # (B, num_classes, H, W)
        u1 = self.num_classes / S1  # (B, 1, H, W)
        b1 = b11 * b12_1 + u1 * b12_1  # (B, num_classes, H, W)
        kappa1 = torch.sum(b1, dim=1, keepdim=True)  # (B, 1, H, W)
        kappa1 = kappa1 + epsilon
        b_new1 = (1 - u1) * b1 / kappa1  # (B, num_classes, H, W)
        alpha_new1 = S1 * b_new1 + 1  # (B, num_classes, H, W)

        # 计算第二个模态的 Dirichlet 分布参数
        alpha21 = e2 + 1  # (B, num_classes, H, W)
        S2 = torch.sum(alpha21, dim=1, keepdim=True)  # (B, 1, H, W)
        S2 = S2 + epsilon
        b21 = e2 / S2  # (B, num_classes, H, W)
        u2 = self.num_classes / S2  # (B, 1, H, W)
        b2 = b21 * b12_2 + u2 * b12_2  # (B, num_classes, H, W)
        kappa2 = torch.sum(b2, dim=1, keepdim=True)  # (B, 1, H, W)
        kappa2 = kappa2 + epsilon
        b_new2 = (1 - u2) * b2 / kappa2  # (B, num_classes, H, W)
        alpha_new2 = S2 * b_new2 + 1  # (B, num_classes, H, W)

        # 融合两个模态的 Dirichlet 分布
        b = b_new1 * b_new2 + b_new1 * u2 + b_new2 * u1  # (B, num_classes, H, W)
        u = u1 * u2  # (B, 1, H, W)
        kappa = torch.sum(b, dim=1, keepdim=True) + u  # (B, 1, H, W)
        kappa = kappa + epsilon
        b_new = b / kappa  # (B, num_classes, H, W)
        u_new = u / kappa  # (B, 1, H, W)
        u_new = u_new + epsilon  # 防止除零
        S = self.num_classes / u_new  # (B, 1, H, W)
        alpha_new = S * b_new + 1  # (B, num_classes, H, W)

        # 计算期望输出 m
        m = alpha_new / torch.sum(alpha_new, dim=1, keepdim=True)  # (B, num_classes, H, W)

        return m, alpha_new  # 返回输出和融合后的 Dirichlet 分布参数


