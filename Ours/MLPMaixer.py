import torch
import torch.nn as nn
from HGNN import HGNN
import hypergraph_utils_gpu as hgut_gpu
from visual_data import *
from matplotlib import pyplot as plt
from FusionBy3D import MultiViewFusion
import numpy as np
from StyleGAN import*
from selfattn import SelfAttention
from GhostV3 import *
from PIL import Image

model_urls = {
    "starnet_s1": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s1.pth.tar",
    "starnet_s2": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s2.pth.tar",
    "starnet_s3": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s3.pth.tar",
    "starnet_s4": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s4.pth.tar",
}
import cv2

class MlpBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_dim):
        super(MlpBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim)
        )

    def forward(self, x):
        return self.mlp(x) ##每一次的全连接中间有个隐藏层，经过隐藏层再回到与输入一致的维度大小。


class MixerBlock(nn.Module):
    def __init__(self, num_tokens, hidden_dim, tokens_mlp_dim, channels_mlp_dim):
        super(MixerBlock, self).__init__()
        self.ln_token = nn.LayerNorm(hidden_dim)
        self.token_mix = MlpBlock(num_tokens, tokens_mlp_dim)
        self.ln_channel = nn.LayerNorm(hidden_dim)
        self.channel_mix = MlpBlock(hidden_dim, channels_mlp_dim)

    def forward(self, x):
        out = self.ln_token(x).transpose(1, 2)
        x = x + self.token_mix(out).transpose(1, 2) ##先进行一次token-mixing MLP
        out = self.ln_channel(x)
        x = x + self.channel_mix(out) ##再进行一次channel-mixing MLP
        return x


class MlpMixer(nn.Module):
    ##描述整体的MLP-Mixer框架
    def __init__(self, num_classes, num_blocks, patch_size, hidden_dim, tokens_mlp_dim, channels_mlp_dim, image_size=448):
        super(MlpMixer, self).__init__()
        num_tokens = (image_size // patch_size)**2

        self.patch_emb = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.mlp = nn.Sequential(*[MixerBlock(num_tokens, hidden_dim, tokens_mlp_dim, channels_mlp_dim) for _ in range(num_blocks)])
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.patch_emb(x)
        x = x.flatten(2).transpose(1, 2) ##制造生成分patch的input
        x = self.mlp(x) #进行n次mlp layer
        x = self.ln(x)
        x = x.mean(dim=1) #average pooling的操作
        x = self.fc(x)  #全连接至分类数
        return x


def mixer_s32(num_classes=1000, **kwargs):
    return MlpMixer(num_classes, 4, 32, 512, 256, 2048, **kwargs)


class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)



class StarNet(nn.Module):
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1000, **kwargs):
        super(StarNet, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.autoencoder=nn.Linear(36,512)
        self.autodecoder=nn.Linear(512,32)
        self.selfattn1 = SelfAttention(512,head_dim=256)
        # self.selfattn2 = SelfAttention(128,head_dim=64)
        # self.to512 = nn.Linear(36,512)
        # self.gen = StyleGAN3Generator()
        # self.head = nn.Linear(self.in_channel, num_classes)
        self.head2=nn.Linear(32,num_classes)
        self.Conv_448To224=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=5,stride=3,padding=2)
        self.Conv_224To128=nn.Conv2d(in_channels=64,out_channels=32,kernel_size=5,stride=3,padding=2)

        self.relu=nn.ReLU()
        self.HyperG = HGNN(50 * 50, 32, 512, 0.2)
        # self.apply(self._init_weights)
        # self.hyper_cewei_wowei = HGNN(448, 448, 672, 0.2)
        # self.conv_cewei_wowei = nn.Conv2d(1, 1, 3, (2, 1), 1)
        # self.hyper_cewei_liwei = HGNN(448, 448, 672, 0.2)
        # self.conv_cewei_liwei = nn.Conv2d(1, 1, 3, (2, 1), 1)
        # self.hyper_wowei_liwei = HGNN(448, 448, 672, 0.2)
        # self.conv_wowei_liwei = nn.Conv2d(1, 1, 3, (2, 1), 1)
        batch_size=8
        # 相机内参（假设所有相机的内参相同）
        fx = 224.0  # 焦距（x方向）
        fy = 224.0  # 焦距（y方向）
        cx = 224.0  # 图像中心（x方向）
        cy = 224.0  # 图像中心（y方向）
        # 创建相机参数
        self.K = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=torch.float32).repeat(batch_size, 1, 1)

        self.RT_front = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ], dtype=torch.float32).repeat(batch_size, 1, 1)

        self.RT_back = torch.tensor([
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0]
        ], dtype=torch.float32).repeat(batch_size, 1, 1)

        self.RT_side = torch.tensor([
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0]
        ], dtype=torch.float32).repeat(batch_size, 1, 1)
        # 创建模型实例,in_channels为所有但视角图片通道数
        # self.fusion_model = MultiViewFusion(in_channels=1, out_channels=64, H=448, W=448)
        self.MLP=mixer_s32(32)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            # trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x,x_f):

        # x=self.fusion_model(x[:,2:],x[:,1:2],x[:,0:1],self.K, self.K, self.K, self.RT_front, self.RT_back,self.RT_side)
        # x = self.relu(self.Conv_448To224(x))
        # x = self.relu(self.Conv_224To128(x))
        x_f = self.relu(self.autoencoder(x_f))
        x_f = self.selfattn1(x_f)
        x_f = self.relu(self.autodecoder(x_f))
        # x = self.Get_HyperGraph(x)
        # print(x.shape)
        x=self.MLP(x)
        x=x+x_f
        # x=+x_f
        # print(x.shape)
        x=self.head2(x)
        # print(x.shape)
        return x
        # return x


# @register_model
def starnet_s1(pretrained=False, **kwargs):
    model = StarNet(24, [2, 2, 8, 3], **kwargs)
    if pretrained:
        url = model_urls['starnet_s1']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    return model


# @register_model
def starnet_s2(pretrained=False, **kwargs):
    model = StarNet(32, [1, 2, 6, 2], **kwargs)
    if pretrained:
        url = model_urls['starnet_s2']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    return model


# @register_model
def starnet_s3(pretrained=False, **kwargs):
    model = StarNet(32, [2, 2, 8, 4], **kwargs)
    if pretrained:
        url = model_urls['starnet_s3']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    return model


# @register_model
def starnet_s4(pretrained=False, **kwargs):
    model = StarNet(32, [3, 3, 12, 5], **kwargs)
    if pretrained:
        url = model_urls['starnet_s4']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    return model


# very small networks #
# @register_model
def starnet_s050(pretrained=False, **kwargs):
    return StarNet(16, [1, 2, 2, 1], 2, **kwargs)


# @register_model
def starnet_s100(pretrained=False, **kwargs):
    return StarNet(20, [1, 2, 4, 1], 4, **kwargs)


# @register_model
def starnet_s150(pretrained=False, **kwargs):
    return StarNet(24, [1, 2, 4, 2], 3, **kwargs)


def Test_Model():
    model = starnet_s050(num_classes=2).cuda()
    Tensor1 = torch.rand(8, 3,448,448).cuda()
    Tensor2 = torch.rand(8, 36).cuda()
    A ,_= model(Tensor1,Tensor2)
    print(A.shape)
# Test_Model()