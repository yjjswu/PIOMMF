# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import torch
# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score
# import numpy as np
# from DataOfClass import Dataset2
# from StarNet2 import *
#
# class DetectionModel(nn.Module):
#     def __init__(self, num_classes):
#         super(DetectionModel, self).__init__()
#         self.backbone = starnet_s050(num_classes=num_classes)
#
#     def forward(self, x, x_f):
#         x = self.backbone(x, x_f)
#         return x
#
#
# def test_model_with_feature_dim_drop(model, dataloader, num_classes, feature_dim, drop_dim):
#     """特征维度缺失分析"""
#
#     def drop_feature_dim(features):
#         # 创建一个掩码，将指定维度的特征置为0
#         mask = torch.ones_like(features)
#         mask[:, drop_dim] = 0
#         return features * mask
#
#     print(f"\n=== 特征维度{drop_dim}缺失测试 ===")
#     return base_test_model(model, dataloader, num_classes,
#                            feature_processor=drop_feature_dim)
#
#
# def base_test_model(model, dataloader, num_classes, **kwargs):
#     # model.eval()
#     all_preds = []
#     all_targets = []
#
#     with torch.no_grad():
#         for images, targets, features in dataloader:
#             if 'image_processor' in kwargs:
#                 images = kwargs['image_processor'](images)
#             if 'feature_processor' in kwargs:
#                 features = kwargs['feature_processor'](features)
#
#             images = images.cuda()
#             targets = targets.cuda()
#             features = features.cuda().type(torch.float32)
#
#             outputs = model(images, features)
#             _, preds = torch.max(outputs, 1)
#             all_preds.extend(preds.cpu().numpy())
#             all_targets.extend(targets.cpu().numpy())
#
#     accuracy = accuracy_score(all_targets, all_preds)
#     return accuracy
#
#
# def plot_sensitivity(results_dict, title):
#     plt.figure(figsize=(10, 6))
#     plt.rcParams['font.family'] = ['SimHei']  ## win
#     plt.rcParams["axes.unicode_minus"] = False
#     for dim, values in results_dict.items():
#         x = list(values.keys())
#         y = [v * 100 for v in values.values()]
#         plt.plot(x, y, marker='o', label=dim)
#     plt.ylim(0, 100.0)
#     plt.title(f"模型敏感性分析 - {title}")
#     plt.xlabel("特征维度")
#     plt.ylabel("准确率 (%)")
#     plt.grid(True)
#     plt.legend()
#     plt.show()
#
#
# if __name__ == "__main__":
#     model = torch.load('./weights/0516_2.pt', map_location='cuda').cuda()
#     dataloader = Dataset2  # 您的数据加载器
#     results = {'特征维度缺失': {}}
#
#     # 获取特征向量的维度
#     feature_dim = 36  # 假设特征向量的维度为128
#     for drop_dim in range(feature_dim):
#         acc = test_model_with_feature_dim_drop(model, dataloader, 4, feature_dim, drop_dim)
#         results['特征维度缺失'][str(drop_dim)] = acc
#
#     plot_sensitivity(results, "特征维度缺失敏感性分析")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
from DataOfClass import Dataset2
from StarNet2 import *

class DetectionModel(nn.Module):
    def __init__(self, num_classes):
        super(DetectionModel, self).__init__()
        self.backbone = starnet_s050(num_classes=num_classes)

    def forward(self, x, x_f):
        x = self.backbone(x, x_f)
        return x

def base_test_model(model, dataloader, num_classes, **kwargs):
    # model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets, features in dataloader:
            if 'image_processor' in kwargs:
                images = kwargs['image_processor'](images)
            if 'feature_processor' in kwargs:
                features = kwargs['feature_processor'](features)
            for i in range(10):
                images = images.cuda()+torch.randn_like(images).cuda()*0.28
                targets = targets.cuda()
                features = features.cuda().type(torch.float32)

                outputs = model(images, features)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_preds)
    return accuracy

def test_model_with_feature_dim_drop(model, dataloader, num_classes, feature_dim, drop_dim):
    """特征维度缺失分析"""

    def drop_feature_dim(images):
        # 创建一个掩码，将指定维度的特征置为0
        mask = torch.ones_like(images)
        mask[:, drop_dim,:,:] = 0
        return images * mask

    print(f"\n=== 特征维度{drop_dim}缺失测试 ===")
    return base_test_model(model, dataloader, num_classes,
                           image_processor=drop_feature_dim)

def plot_feature_importance(importances, feature_dim):
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.family'] = ['SimHei']  ## win
    plt.rcParams["axes.unicode_minus"] = False

    indices = np.argsort(importances)[::-1]
    plt.bar(range(feature_dim), importances[indices], align='center', alpha=0.8)
    plt.plot(range(feature_dim), importances[indices])
    plt.scatter(range(feature_dim), importances[indices])
    plt.xticks(range(feature_dim), indices)
    plt.xlabel("特征维度")
    plt.ylabel("重要性")
    plt.title("特征重要性评估")
    plt.show()

if __name__ == "__main__":
    model = torch.load('./weights/0516_2.pt', map_location='cuda').cuda()
    dataloader = Dataset2  # 您的数据加载器
    feature_dim = 3  # 假设特征向量的维度为36

    # 初始化特征重要性数组
    importances = np.zeros(feature_dim)

    # 计算每个特征维度的重要性
    for drop_dim in range(feature_dim):
        acc = test_model_with_feature_dim_drop(model, dataloader, 4, feature_dim, drop_dim)
        importances[drop_dim] = 1-acc

    # 可视化特征重要性
    plot_feature_importance(importances, feature_dim)