import torch
from ptflops import get_model_complexity_info
import torchvision.models as models
from StarNet2 import *
from torch import nn
from thop import profile
class DetectionModel(nn.Module):
    def __init__(self, num_classes):
        super(DetectionModel, self).__init__()
        # self.backbone = resnet50(pretrained=False)
        # self.backbone = nn.Sequential(*list(self.backbone.children())[:-4])  # 去掉最后的全连接层和池化层
        self.backbone = starnet_s050(num_classes=num_classes)

    def forward(self, x,x_f):
        x = self.backbone(x,x_f)
        return x


# 创建模型对象
model = torch.load("./weights/MEd.pt").cuda()
print(model)
def count_all_parameters(model):
    return sum(p.numel() for p in model.parameters())

# # 使用示例：
# # 替换为你的模型
# total_params = count_all_parameters(model)
# print(f"Total parameters: {total_params}")

# 创建一个随机输入张量（假设输入形状为3通道，224x224）
input_tensor = torch.randn(1, 3, 448, 448).cuda()
feat= torch.randn(1,36).float().cuda()

# 使用thop库计算FLOPs和参数量
flops, params = profile(model, inputs=(input_tensor,feat))

print(f"GFLOPs: {flops / 1e9:.2f}")
print(f"Parameters: {params / 1e6:.2f}M")