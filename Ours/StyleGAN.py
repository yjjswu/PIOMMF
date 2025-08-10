import torch
import torch.nn as nn
import torch.nn.functional as F

# #------------------------------
# # 实现缺失的upsample_2d函数
# #------------------------------
# def upsample_2d(x, kernel, gain=1):
#     # 创建2D滤波器核
#     kernel = kernel / kernel.sum()  # 归一化
#     kernel_2d = torch.outer(kernel, kernel) * gain  # 外积创建2D核
    
#     # 准备转置卷积参数
#     C = x.size(1)  # 输入通道数
#     kernel_size = kernel_2d.size(0)
#     padding = kernel_size // 2
    
#     # 创建深度转置卷积
#     weight = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(C, 1, 1, 1)
#     return F.conv_transpose2d(
#         x, weight,
#         stride=2,          # 上采样因子2
#         padding=padding,   # 保持尺寸对齐
#         groups=C           # 深度卷积
#     )

# #------------------------------
# # 修改后的上采样模块
# #------------------------------
# class UpFIRESample(nn.Module):
#     def __init__(self, resample_kernel):
#         super().__init__()
#         kernel_tensor = torch.tensor(resample_kernel, dtype=torch.float32)
#         self.register_buffer('resample_kernel', kernel_tensor)
    
#     def forward(self, x):
#         return upsample_2d(x, self.resample_kernel, gain=1)

#------------------------------
# 修正后的调制卷积
#------------------------------
class ModulatedConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, style_dim, demodulate=True):
        super().__init__()
        self.out_ch = out_ch
        self.kernel = kernel
        self.pad = kernel // 2
        
        # 初始化权重
        self.weight = nn.Parameter(torch.randn(1, out_ch, in_ch, kernel, kernel))
        self.modulation = nn.Linear(style_dim, in_ch)
        self.demodulate = demodulate

    def forward(self, x, style):
        b, c, h, w = x.shape
        
        # 样式调制
        style = self.modulation(style).view(b, 1, c, 1, 1)
        weight = self.weight * style
        
        # 解调
        if self.demodulate:
            d = torch.rsqrt(weight.pow(2).sum([2,3,4]) + 1e-8)
            weight = weight * d.view(b, self.out_ch, 1, 1, 1)
            
        # 重塑权重进行分组卷积
        weight = weight.view(b*self.out_ch, c, self.kernel, self.kernel)
        x = x.view(1, b*c, h, w)
        x = F.conv2d(x, weight, padding=self.pad, groups=b)
        return x.view(b, self.out_ch, h, w)  # 修正：保持尺寸不变

# #------------------------------
# # 合成块（包含上采样）
# #------------------------------
# class SynthesisBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, style_dim, kernel):
#         super().__init__()
#         self.upsample = UpFIRESample([1.0, 3.0, 3.0, 1.0])
#         self.conv1 = ModulatedConv2d(in_ch, out_ch, kernel, style_dim)
#         self.conv2 = ModulatedConv2d(out_ch, out_ch, kernel, style_dim)

#     def forward(self, x, style):
#         x = self.upsample(x)   # 上采样到2倍分辨率
#         x = self.conv1(x, style)
#         x = self.conv2(x, style)
#         return x

# #------------------------------
# # 完整生成器
# #------------------------------
# class StyleGAN3Generator(nn.Module):
#     def __init__(self, style_dim=512):
#         super().__init__()
#         # 样式映射网络
#         self.mapping = nn.Sequential(
#             nn.Linear(style_dim, style_dim),
#             nn.LeakyReLU(0.2),
#             nn.Linear(style_dim, style_dim)
#         )
        
#         # 合成网络
#         self.blocks = nn.ModuleList([
#             SynthesisBlock(512, 512, style_dim, 3),
#             SynthesisBlock(512, 256, style_dim, 3),
#             SynthesisBlock(256, 128, style_dim, 3),
#             SynthesisBlock(128, 64, style_dim, 3),
#             SynthesisBlock(64, 32, style_dim, 3),
#             SynthesisBlock(32,24,style_dim,3),
#             SynthesisBlock(24,12,style_dim,3)
#         ])
        
#         # 最终RGB转换
#         self.to_rgb = nn.Conv2d(12, 3, kernel_size=1)

#     def forward(self, z):
#         # 生成样式向量
#         style = self.mapping(z)
        
#         # 初始4x4特征图
#         x = torch.randn(z.size(0), 512, 4, 4, device=z.device)
        
#         # 通过所有合成块
#         for block in self.blocks:
#             x = block(x, style)
#             print(x.shape)
#         # 转换为RGB图像
#         return torch.tanh(self.to_rgb(x))

import torch
import torch.nn as nn
import torch.nn.functional as F

#------------------------------
# 自定义尺寸上采样模块
#------------------------------
class AdaptiveUpsample(nn.Module):
    def __init__(self, target_size):
        super().__init__()
        self.target_size = target_size
    
    def forward(self, x):
        # 使用双线性插值精确控制输出尺寸
        return F.interpolate(
            x, size=self.target_size,
            mode='bilinear', align_corners=False
        )

#------------------------------
# 带尺寸控制的合成块
#------------------------------
class SynthesisBlock(nn.Module):
    def __init__(self, in_ch, out_ch, style_dim, kernel, target_size):
        super().__init__()
        self.upsample = AdaptiveUpsample(target_size)
        self.conv1 = ModulatedConv2d(in_ch, out_ch, kernel, style_dim)
        self.conv2 = ModulatedConv2d(out_ch, out_ch, kernel, style_dim)
        
        # 保持尺寸的卷积配置
        self.conv1.pad = (kernel-1)//2  # 自动计算padding
        self.conv2.pad = (kernel-1)//2

    def forward(self, x, style):
        x = self.upsample(x)    # 精确尺寸上采样
        x = self.conv1(x, style)
        x = self.conv2(x, style)
        return x

#------------------------------
# 修正后的生成器架构
#------------------------------
class StyleGAN3Generator(nn.Module):
    def __init__(self, style_dim=512):
        super().__init__()
        # 样式映射网络保持不变
        self.mapping = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(style_dim, style_dim)
        )
        
        # 定义分辨率演进路径：4→6→18
        self.blocks = nn.ModuleList([
            SynthesisBlock(512, 512, style_dim, 3, target_size=(14,14)),  # 4→6
            SynthesisBlock(512, 256, style_dim, 3, target_size=(28,28)),# 6→18
            SynthesisBlock(256, 128, style_dim, 3, target_size=(56,56)),# 6→18
            SynthesisBlock(128, 64, style_dim, 3, target_size=(112,112)),# 6→18
            SynthesisBlock(64, 32, style_dim, 3, target_size=(224,224)),# 6→18
            SynthesisBlock(32, 12, style_dim, 3, target_size=(448,448)),# 6→18
            # SynthesisBlock(12, 6, style_dim, 3, target_size=(512,512)),# 6→18
            # 后续层按需添加...
        ])
        
        # 初始化噪声特征图
        self.init_constant = nn.Parameter(torch.randn(1, 512, 7, 7))
        
        # 最终转换层
        self.to_rgb = nn.Sequential(
            nn.Conv2d(12, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 1)
        )

    def forward(self, z):
        style = self.mapping(z)
        
        # 初始特征图（批量复制）
        x = self.init_constant.repeat(z.size(0), 1, 1, 1)
        
        # 通过合成网络
        for block in self.blocks:
            x = block(x, style)
        
        # 输出RGB
        return torch.tanh(self.to_rgb(x))

# # 测试代码
# if __name__ == "__main__":
#     model = StyleGAN3Generator()
#     z = torch.randn(2, 512)  # 匹配style_dim=512
#     output = model(z)
#     print(output.shape)  # 应该输出 torch.Size([2, 3, 128, 128])

if __name__ == "__main__":
    # 创建模型
    model = StyleGAN3Generator()
    
    # 测试输入
    z = torch.randn(2, 512)
    output = model(z)
    
    # 验证尺寸演进
    print("输出尺寸:", output.shape)  # 预期输出: torch.Size([2, 3, 18, 18])
    
    # 可视化分辨率变化
    print("\n特征图尺寸演进：")
    x = model.init_constant
    print("初始尺寸:", x.shape[2:])
    for i, block in enumerate(model.blocks):
        x = block(x, torch.randn(1,512))
        print(f"第{i+1}层后尺寸:", x.shape[2:])