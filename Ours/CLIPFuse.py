import torch
import torch.nn as nn
from PIL import Image
import clip
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import os
from torchvision.transforms import transforms
import pandas as pd
print(clip.available_models())
# 1. 加载预训练的CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model, preprocess = clip.load("ViT-B/32", device=device)
print(next(model.parameters()).device)
ReadFile=pd.read_excel('./All.xlsx',sheet_name='Sheet1')
# print(ReadFile.shape)
Features=np.array(ReadFile[ReadFile.columns[4:]])
Dict_tupianzu_to_label={}
Dict_tupianzu_to_feature={}
for i in range(len(ReadFile)):
    Dict_tupianzu_to_label[ReadFile['图片组'][i]]=ReadFile['主要诊断'][i]
    Dict_tupianzu_to_feature[ReadFile['图片组'][i]]=Features[i,:]

Trans1 = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize([224, 224]),
    ]
)
# 冻结CLIP模型的参数
for param in model.parameters():
    param.requires_grad = True


# 2. 自定义数据集类
import random
class CLIPDataset(Dataset):
    def __init__(self, root:str,IsTrain:bool,preprocess):
        self.PathsOfDirs = []
        self.root = root
        self.list_dir1 = os.listdir(os.path.join(self.root, 'Baoshou'))
        self.list_index1 = list(range(len(self.list_dir1)))
        random.shuffle(self.list_index1)
        self.list_dir1 = [self.list_dir1[index] for index in self.list_index1]
        for item in self.list_dir1:
            if len(os.listdir(root + "/" + "Baoshou/" + item)) >= 3:
                self.PathsOfDirs.append(os.path.join(self.root, "Baoshou", item))
        if (IsTrain):
            self.PathsOfDirs = self.PathsOfDirs[0:int(len(self.PathsOfDirs) * 0.8)]
        else:
            self.PathsOfDirs = self.PathsOfDirs[int(len(self.PathsOfDirs) * 0.8):]
        self.target = [0] * len(self.PathsOfDirs)
        print(len(self.PathsOfDirs))

        Temp_dirs = os.listdir(os.path.join(self.root, "Shoushu"))
        list_index2 = list(range(len(Temp_dirs)))
        random.shuffle(list_index2)
        Temp_dirs = [Temp_dirs[index] for index in list_index2]
        if (IsTrain):
            Temp_dirs = Temp_dirs[0:int(len(Temp_dirs) * 0.8)]
        else:
            Temp_dirs = Temp_dirs[int(len(Temp_dirs) * 0.8):]
        print(len(Temp_dirs))
        Conut = 0
        for item in Temp_dirs:
            if len(os.listdir(root + "/" + "Shoushu/" + item)) >= 3:
                self.PathsOfDirs.append(os.path.join(self.root, "Shoushu", item))
                Conut += 1
        self.target.extend([1] * Conut)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        img_paths = os.listdir(self.PathsOfDirs[index])
        img_paths.sort(key=lambda x: x[-6])
        # print(img_paths)
        try:
            img_path1 = os.path.join(self.root, self.PathsOfDirs[index], img_paths[0])
            img_path2 = os.path.join(self.root, self.PathsOfDirs[index], img_paths[1])
            img_path3 = os.path.join(self.root, self.PathsOfDirs[index], img_paths[2])
            i1 = Image.open(img_path1, mode='r').convert('L')
            i2 = Image.open(img_path2, mode='r').convert('L')
            i3 = Image.open(img_path3, mode='r').convert('L')
            img1 = Trans1(i1)
            img2 = Trans1(i2)
            img3 = Trans1(i3)
            # print("123",img2.shape)
        except:
            print(self.PathsOfDirs[index])
            print(os.listdir(self.PathsOfDirs[index]))
            return "x"
        img = torch.concat((img1, img2, img3))
        # image = self.preprocess(img)
        return img, Dict_tupianzu_to_label[self.PathsOfDirs[index][self.PathsOfDirs[index].rfind('\\')+1:]],Dict_tupianzu_to_feature[self.PathsOfDirs[index][self.PathsOfDirs[index].rfind('\\')+1:]]
Dataset1 = DataLoader(CLIPDataset(r"D:\\ZJF\pythongrams\coding\Datas",True,preprocess), batch_size=16*6, shuffle=True)
Dataset2 = DataLoader(CLIPDataset(r"D:\\ZJF\pythongrams\coding\Datas",False,preprocess), batch_size=16*6, shuffle=True)


# 3. 特征融合分类模型
class CLIPFusionClassifier(nn.Module):
    def __init__(self, clip_model, num_classes, feature_dim=512):
        super().__init__()
        self.clip_model = clip_model
        self.fc = nn.Linear(feature_dim * 2, num_classes)  # 融合图像和文本特征
        self.fc_text=nn.Linear(36,77,dtype=torch.float32)
    def forward(self, images, texts):
        # 提取图像特征
        image_features = self.clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # 提取文本特征
        # tokenized_text = clip.tokenize(texts, truncate=True).to(device)
        # print(texts.dtype)
        # print(self.fc_text.weight.dtype)
        texts=self.fc_text(texts)
        tokenized_text=(texts*100)
        tokenized_text=(((-1)*torch.min(tokenized_text)+tokenized_text)%49408).long()
        text_features = self.clip_model.encode_text(tokenized_text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 融合特征
        fused_features = torch.cat([image_features, text_features], dim=-1).float()

        # 分类
        logits = self.fc(fused_features)
        return logits


# 4. 训练函数
def train_model(model, dataloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images,  labels, texts, in dataloader:
            images = images.to(device)
            labels = labels.to(device).long()
            texts = texts.to(device).float()
            optimizer.zero_grad()

            outputs = model(images, texts)
            print(outputs)
            loss = criterion(outputs, labels)
            print(loss.item())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")


# 5. 评估函数
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, texts, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images, texts)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


# 6. 主函数
def main():


    # 创建数据集和数据加载器
    # train_dataset = Dataset1
    # test_dataset = Dataset2

    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 初始化模型
    num_classes = 4
    fusion_model = CLIPFusionClassifier(model, num_classes).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=0.001)

    # 训练模型
    train_model(fusion_model, Dataset1, criterion, optimizer, epochs=10)

    # 评估模型
    evaluate_model(fusion_model, Dataset2)


if __name__ == "__main__":
    main()
    # for i,(img,label,text) in enumerate(Dataset1):
    #     print(img)
    # print("pass")