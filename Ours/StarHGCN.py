"""
Implementation of Prof-of-Concept Network: StarNet.

We make StarNet as simple as possible [to show the key contribution of element-wise multiplication]:
    - like NO layer-scale in network design,
    - and NO EMA during training,
    - which would improve the performance further.

Created by: Xu Ma (Email: ma.xu1@northeastern.edu)
Modified Date: Mar/29/2024
"""
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
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
class GetImageFromMid():
    def __init__(self):
        self.index=0
    def CheckImage(self,image_ten:torch.Tensor):
        # 将tensor转换为numpy数组
        image_tenx = image_ten.detach().cpu().numpy()
        # 遍历每个通道
        for i in range(image_tenx.shape[0]):
            # 提取单个通道的特征图
            single_channel = image_tenx[i, :, :]
            # 将特征图的值归一化到0-255
            single_channel = (single_channel - np.min(single_channel)) / (
                        np.max(single_channel) - np.min(single_channel)) * 255
            single_channel = single_channel.astype(np.uint8)
            heatmap = cv2.applyColorMap(single_channel, cv2.COLORMAP_HSV)
            heatmap = cv2.rotate(heatmap,cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(f'./DeepFeat/channel_{i}_color.png', heatmap)  # 保存图像
            # 保存图像
            # img.save(f'channel_{i}_color.png')



        # # 获取通道数、高度和宽度
        # channels, heighttimeswidth = image_ten.shape
        # # 初始化一个空列表，用于存储展平后的向量
        # flattened_vectors = []
        # # 遍历每个通道
        # for i in range(channels):
        #     # 提取单个通道的特征图
        #     single_channel = image_ten[i, :]
        #     # 将特征图展平为一维向量
        #     # flattened_vector = single_channel.flatten()
        #     # print(flattened_vector.shape)
        #     # 将展平后的向量添加到列表中
        #     flattened_vectors.append(single_channel)
        # # 将所有展平后的向量拼接为一个矩阵
        # concatenated_matrix = np.concatenate(flattened_vectors, axis=0)
        # # 将矩阵转换为图像
        # concatenated_matrix = (concatenated_matrix - np.min(concatenated_matrix)) / (np.max(concatenated_matrix) - np.min(concatenated_matrix)) * 255
        # concatenated_matrix = concatenated_matrix.astype(np.uint8)
        # print(concatenated_matrix.shape)
        # img = Image.fromarray(concatenated_matrix.reshape(int(np.sqrt(len(concatenated_matrix))), -1), "L")
        # img = img.rotate(90)
        # 保存图像
        # 将tensor转换为numpy数组
        # image_ten = torch.flatten(image_ten, 1)
        # image_ten = image_ten.detach().cpu().numpy()
        # # print(image_ten.shape)
        # image_ten = (image_ten - np.min(image_ten)) / (np.max(image_ten) - np.min(image_ten)) * 255
        # image_ten = image_ten.astype(np.uint8)
        # # 使用OpenCV生成热力图
        # heatmap = cv2.applyColorMap(image_ten, cv2.COLORMAP_HSV)
        # cv2.imwrite(f'./MidFeat/heatmap_cool_{self.index}.png', heatmap)  # 保存图像
        # self.index+=1
        # cv2.imshow('Heatmap', heatmap)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
GetImage=GetImageFromMid()
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        # 定义卷积层
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        # 条件性地添加批归一化层
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        # 条件性地添加ReLU激活函数
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)  # 应用卷积
        if self.bn is not None:
            x = self.bn(x)  # 应用批归一化
        if self.relu is not None:
            x = self.relu(x)  # 应用ReLU
        return x


# 定义ZPool模块，结合最大池化和平均池化结果
class ZPool(nn.Module):
    def forward(self, x):
        # 结合最大值和平均值
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


# 定义注意力门，用于根据输入特征生成注意力权重
class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7  # 设定卷积核大小
        self.compress = ZPool()  # 使用ZPool模块
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)  # 通过卷积调整通道数

    def forward(self, x):
        x_compress = self.compress(x)  # 应用ZPool
        x_out = self.conv(x_compress)  # 通过卷积生成注意力权重
        scale = torch.sigmoid_(x_out)  # 应用Sigmoid激活
        return x * scale  # 将注意力权重乘以原始特征


# 定义TripletAttention模块，结合了三种不同方向的注意力门
class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()  # 定义宽度方向的注意力门
        self.hc = AttentionGate()  # 定义高度方向的注意力门
        self.no_spatial = no_spatial  # 是否忽略空间注意力
        if not no_spatial:
            self.hw = AttentionGate()  # 定义空间方向的注意力门

    def forward(self, x):
        # 应用注意力门并结合结果
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()  # 转置以应用宽度方向的注意力
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()  # 还原转置
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()  # 转置以应用高度方向的注意力
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()  # 还原转置
        if not self.no_spatial:
            x_out = self.hw(x)  # 应用空间注意力
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)  # 结合三个方向的结果
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)  # 结合两个方向的结果（如果no_spatial为True）
        return x_out


class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.Out1 = TripletAttention()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        x = self.Out1(x)
        return x
import torchvision
from MedVIT import *
class StarNet(nn.Module):
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1000, **kwargs):
        super(StarNet, self).__init__()
        # self.avgpool = nn.AdaptiveAvgPool2d(2)
        self.autoencoder=nn.Linear(36,512)
        self.autodecoder=nn.Linear(512,32)
        self.selfattn1 = SelfAttention(512,head_dim=256)
        # self.selfattn2 = SelfAttention(128,head_dim=64)
        # self.to512 = nn.Linear(36,512)
        # self.gen = StyleGAN3Generator()
        # self.head = nn.Linear(self.in_channel, num_classes)
        self.head2=nn.Linear(32,num_classes)
        self.Conv_448To224=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=5,stride=3,padding=2)
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


    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def Get_HyperGraph_2(self, img):
        # 预分配列表用于存储结果
        List_cewei_wowei = []
        List_cewei_liwei = []
        List_wowei_liwei = []

        # 将img转换为GPU张量
        img = img.cuda()

        for i in range(img.size(0)):
            imgi = img[i]
            # 重塑图像通道
            img_cewei = imgi[0].view(448, 448)
            img_wowei = imgi[1].view(448, 448)
            img_liwei = imgi[2].view(448, 448)

            # 在GPU上拼接特征
            ft_cewei_wowei = torch.cat((img_cewei, img_wowei), dim=0).cuda()
            ft_cewei_liwei = torch.cat((img_cewei, img_liwei), dim=0).cuda()
            ft_wowei_liwei = torch.cat((img_wowei, img_liwei), dim=0).cuda()

            # 构造超图并生成图矩阵
            H1 = hgut_gpu.construct_H_with_KNN(ft_cewei_wowei, K_neigs=[5],
                                                split_diff_scale=False,
                                                is_probH=True, m_prob=1).cuda()
            G1 = hgut_gpu.generate_G_from_H(H1).cuda()
            ft_cewei_wowei = self.hyper_cewei_wowei(ft_cewei_wowei, G1)
            List_cewei_wowei.append(ft_cewei_wowei.view(1, 1, 448 * 2, 448))

            H2 = hgut_gpu.construct_H_with_KNN(ft_cewei_liwei, K_neigs=[5],
                                                split_diff_scale=False,
                                                is_probH=True, m_prob=1).cuda()
            G2 = hgut_gpu.generate_G_from_H(H2).cuda()
            ft_cewei_liwei = self.hyper_cewei_liwei(ft_cewei_liwei, G2)
            List_cewei_liwei.append(ft_cewei_liwei.view(1, 1, 448 * 2, 448))

            H3 = hgut_gpu.construct_H_with_KNN(ft_wowei_liwei, K_neigs=[5],
                                                split_diff_scale=False,
                                                is_probH=True, m_prob=1).cuda()
            G3 = hgut_gpu.generate_G_from_H(H3).cuda()
            ft_wowei_liwei = self.hyper_wowei_liwei(ft_wowei_liwei, G3)
            List_wowei_liwei.append(ft_wowei_liwei.view(1, 1, 448 * 2, 448))

        # 批量处理卷积操作
        conv_cewei_wowei = self.conv_cewei_wowei(torch.cat(List_cewei_wowei))
        conv_cewei_liwei = self.conv_cewei_liwei(torch.cat(List_cewei_liwei))
        conv_wowei_liwei = self.conv_wowei_liwei(torch.cat(List_wowei_liwei))

        List_w = [conv_cewei_wowei, conv_cewei_liwei, conv_wowei_liwei]

        return List_w

    def Get_HyperGraph(self, img):
        # print("imgs:",img.shape)
        G_Feature = []
        for i in range(len(img)):
            mvcnn_ft = torch.flatten(img[i], 1)
            # print('----',mvcnn_ft.shape)
            H = None
            tmp = hgut_gpu.construct_H_with_KNN(mvcnn_ft, K_neigs=[5],
                                                split_diff_scale=False,
                                                is_probH=True, m_prob=1)
            H = hgut_gpu.hyperedge_concat(H, tmp)
            # print("H device",H.device)
            G = hgut_gpu.generate_G_from_H(H).cuda()
            # print(G.device, mvcnn_ft.device)
            feature = self.HyperG(mvcnn_ft, G)
            # print(feature.shape, "0-----------")
            G_Feature.append(feature.unsqueeze(0))
        # G_Feature=
        return torch.cat(G_Feature, dim=0)
        pass

    def forward(self, x,x_f):

        x=self.fusion_model(x[:,2:],x[:,1:2],x[:,0:1],self.K, self.K, self.K, self.RT_front, self.RT_back,self.RT_side)
        x = self.relu(self.Conv_448To224(x))
        x = self.relu(self.Conv_224To128(x))
        x_f = self.relu(self.autoencoder(x_f))
        x_f = self.selfattn1(x_f)
        x_f = self.relu(self.autodecoder(x_f))
        x = self.Get_HyperGraph(x)
        # print(x.shape)
        x=torch.flatten(torch.mean(x,dim=1),1)+x_f
        # print(x.shape)
        x=self.head2(x)

        return x,x_f
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
    A = model(Tensor1,Tensor2)
    print(A.shape)


if __name__ == '__main__':
    Test_Model()