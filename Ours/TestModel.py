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
from StarNet2 import starnet_s050
from sklearn import manifold
import seaborn as sns
import pandas as pd
import mplcursors
# 定义整体模型
class DetectionModel(nn.Module):
    def __init__(self, num_classes):
        super(DetectionModel, self).__init__()
        # self.backbone = resnet50(pretrained=False)
        # self.backbone = nn.Sequential(*list(self.backbone.children())[:-4])  # 去掉最后的全连接层和池化层
        self.backbone = starnet_s050(num_classes=num_classes)
        # self.conv = nn.Conv2d(192, num_classes + 4, kernel_size=1)  # 1x1卷积层
        # self.conv2 = nn.Conv2d(192,4,kernel_size=1)
        # self.Up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, x_f):
        x = self.backbone(x,x_f)
        # x = self.Up1(x)
        # x = self.conv(x)
        # x1=self.conv2(x)
        return x


# 定义损失函数和优化器
def test_model(model, dataloader, num_classes, num_epochs=400):
    while True:
        # model.eval()  # 设置模型为评估模式
        all_preds = []
        all_targets = []
        all_classvector = []
        Accs=[]
        Recalls=[]
        Precision=[]
        F1score=[]
        with torch.no_grad():  # 禁用梯度计算
            for i in range(4):
                for images, targets, features in dataloader:
                        for i in range(1):
                            images = images.cuda()+torch.randn_like(images).cuda()*0.08
                            targets = targets.cuda()
                            features = features.cuda().type(torch.float32)
                            outputs,classvector= model(images, features)
                            # print(classvector.shape)
                            all_classvector.append(classvector)
                            # 假设输出是概率分布，取最大值的索引作为预测类别
                            _, preds = torch.max(outputs, 1)
                            # 将预测结果和真实标签收集起来
                            all_preds.extend(preds.cpu().numpy())
                            all_targets.extend(targets.cpu().numpy())
                # 计算各类指标
                accuracy = accuracy_score(all_targets, all_preds)
                recall = recall_score(all_targets, all_preds,average='macro')
                precision = precision_score(all_targets, all_preds,average='macro')
                f1 = f1_score(all_targets, all_preds,average='macro')
                Accs.append(accuracy)
                Recalls.append(recall)
                Precision.append(precision)
                F1score.append(f1)
        Accs = np.array(Accs)
        Recalls = np.array(Recalls)
        Precision = np.array(Precision)
        F1score = np.array(F1score)
        print(f'Accuracy: {Accs.mean():.4f},{Accs.std():.4f}')
        print(f'Recall: {Recalls.mean():.4f},{Recalls.std():.4f}')
        print(f'Precision: {Precision.mean():.4f},{Precision.std():.4f}')
        print(f'F1 Score: {F1score.mean():.4f},{F1score.std():.4f}')

        # 如果需要更详细的分类报告，可以使用sklearn的classification_report
        from sklearn.metrics import classification_report
        print("\nClassification Report:")
        print(classification_report(all_targets, all_preds, target_names=[f'Class {i}' for i in range(num_classes)]))

        # 合并所有classvector并转换为numpy数组
        classvector = torch.cat(all_classvector, dim=0).cpu().numpy()
        T_SNE = manifold.TSNE(n_components=3, init='random', random_state=42).fit_transform(classvector)

        # 绘制t-SNE结果，按类别着色
        # plt.figure(figsize=(10, 8))
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown'][:num_classes]
        markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H'][:num_classes]
        fig = plt.figure(figsize=(10, 8))
        plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
        ax = fig.add_subplot(111, projection='3d')
        for i in range(num_classes):
            mask = np.array(all_targets) == i
            ax.scatter(T_SNE[mask, 0], T_SNE[mask, 1],T_SNE[mask,2],
                        c=colors[i], marker=markers[i],
                        label=f'Class {i}', alpha=0.6)

        # plt.title('T-SNE Visualization of Feature Vectors')
        # 设置轴标签
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        plt.legend()
        # plt.grid(True)
        plt.show()
# 假设你已经加载了模型和数据集
num_classes = 4  # 根据你的数据集调整类别数

dataloader = Dataset2  # 你的数据加载器

model = torch.load('./weights/0516_2.pt',weights_only=False).cuda()
print(model)
# model = DetectionModel(num_classes=num_classes).cuda()
test_model(model, dataloader, num_classes, num_epochs=600)