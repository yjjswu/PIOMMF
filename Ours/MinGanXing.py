import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score,accuracy_score,recall_score,precision_score,f1_score
from sklearn.preprocessing import label_binarize
from itertools import cycle, product
import numpy as np
from DataOfClass import Dataset1,Dataset2
# [0.01, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,]).cuda()
Cross1 = nn.CrossEntropyLoss()
from StarNet2 import *
from sklearn import manifold
# 定义整体模型
# 基础测试函数（需稍作修改以支持参数传递）
def base_test_model(model, dataloader, num_classes, **kwargs):
    # model.eval()
    all_preds = []
    all_targets = []
    all_classvector = []

    with torch.no_grad():
        for images, targets, features in dataloader:
            # 允许外部处理图像和特征
            if 'image_processor' in kwargs:
                images = kwargs['image_processor'](images)
            if 'feature_processor' in kwargs:
                features = kwargs['feature_processor'](features)

            images = images.cuda()
            targets = targets.cuda()
            features = features.cuda().type(torch.float32)
            for i in range(5):

                # 处理模型前向传播的特殊情况
                if 'model_processor' in kwargs:
                    outputs = kwargs['model_processor'](model, images)
                else:
                    outputs= model(images,features)

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
    # 计算各类指标
    accuracy = accuracy_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds, average='micro')  # 二分类情况
    precision = precision_score(all_targets, all_preds, average='micro')
    f1 = f1_score(all_targets, all_preds, average='micro')
    # ...（原有评估指标计算和可视化代码保持不变）...
    return accuracy  # 返回主要指标用于分析


# ========== 独立分析函数 ==========

def test_model_with_noise(model, dataloader, num_classes, noise_level=0.1):
    """输入噪声敏感性测试"""

    def add_noise(images):
        return images*(1-noise_level) + torch.randn_like(images) * noise_level

    print(f"\n=== 噪声测试(level={noise_level}) ===")
    return base_test_model(model, dataloader, num_classes,
                           image_processor=add_noise)


def test_model_with_feature_drop(model, dataloader, num_classes, drop_rate=0.3):
    """特征缺失分析"""

    def drop_features(features):
        mask = torch.rand_like(features) < drop_rate
        return features * (~mask).float()

    print(f"\n=== 特征缺失测试(drop_rate={drop_rate}) ===")
    return base_test_model(model, dataloader, num_classes,
                           feature_processor=drop_features)

def test_model_with_threshold(model, dataloader, num_classes, threshold=0.7):
    """分类阈值调整（适用于二分类）"""

    def custom_predict(outputs):
        probs = torch.softmax(outputs, dim=1)
        return (probs[:, 1] > threshold).long()

    print(f"\n=== 阈值调整测试(threshold={threshold}) ===")
    # 需要重写预测逻辑
    acc = base_test_model(model, dataloader, num_classes,
                          predict_processor=custom_predict)
    return acc


def test_model_with_channel_drop(model, dataloader, num_classes, drop_rate=0.2):
    """通道剪裁分析"""
    original_forward = model.forward

    def modified_forward(self, x, x_f):
        x = self.backbone(x, x_f)
        mask = torch.rand(x.size(1)) > drop_rate
        return x * mask[None, :, None, None].cuda()

    # 临时修改前向传播
    model.forward = modified_forward.__get__(model, type(model))
    print(f"\n=== 通道剪裁测试(drop_rate={drop_rate}) ===")
    acc = base_test_model(model, dataloader, num_classes)
    model.forward = original_forward  # 恢复原始实现
    return acc

def plot_sensitivity(results_dict, title):
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.family'] = ['SimHei']  ## win
    plt.rcParams["axes.unicode_minus"] = False
    for dim, values in results_dict.items():
        x = list(values.keys())
        y = [v * 100 for v in values.values()]
        plt.plot(x, y, marker='o', label=dim)
        print(dim,y)
    plt.ylim(0,100.0)
    plt.title(f"模型敏感性分析 - {title}")
    plt.xlabel("干扰占比")
    plt.ylabel("准确率 (%)")
    plt.grid(True)
    plt.legend()
    plt.show()


# 收集测试结果
# results = {
#     "噪声干扰": {0: 0.92, 0.1: 0.89, 0.3: 0.75, 0.5: 0.62},
#     "特征缺失": {0: 0.92, 0.2: 0.85, 0.5: 0.71, 0.8: 0.53},
#     "分辨率变化": {256: 0.92, 224: 0.88, 192: 0.79}
# }

# plot_sensitivity(results, "多维度敏感性分析")
class DetectionModel(nn.Module):
    def __init__(self, num_classes):
        super(DetectionModel, self).__init__()
        self.backbone = starnet_s050(num_classes=num_classes)
    def forward(self, x, x_f):
        x = self.backbone(x,x_f)
        return x
# model

if __name__ == "__main__":
    model = torch.load('./weights/MedVIT_2.pt',weights_only=False).cuda()
    dataloader = Dataset2  # 您的数据加载器
    results={'噪声干扰':{},'特征缺失':{}}
    # 独立测试不同敏感性维度
    noise_levels = list(range(0,100,4))
    for nl in noise_levels:
        acc=test_model_with_noise(model, dataloader, 4, noise_level=nl/100.0)
        results['噪声干扰'][str(nl)]=acc
    # 特征缺失率测试
    drop_rates = list(range(0,100,4))
    for dr in drop_rates:
        acc=test_model_with_feature_drop(model, dataloader, 4, drop_rate=dr/100.0)
        results['特征缺失'][str(dr)] = acc
    plot_sensitivity(results,"多维度敏感性分析")
