import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from torchvision.transforms import transforms
import os
import numpy as np
from PIL import Image
import pandas as pd
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
        transforms.Resize([448, 448]),
    ]
)
Trans2= transforms.Compose(
    [
        transforms.RandomVerticalFlip((0.3)),
        transforms.RandomHorizontalFlip((0.3)),
    ]
)
import random
random.seed(42)
class Dataset():
    def __init__(self, root: str, IsTrain: bool):
        self.PathsOfDirs = []
        self.root = root
        self.list_dir1 = os.listdir(os.path.join(self.root, 'Baoshou'))
        self.list_index1 = list(range(len(self.list_dir1)))
        random.shuffle(self.list_index1) 
        self.list_dir1 = [self.list_dir1[index] for index in self.list_index1]
        for item in self.list_dir1:
            if len(os.listdir(root + "/" + "Baoshou/" + item)) >= 3:
                self.PathsOfDirs.append(os.path.join(self.root, "Baoshou", item))
        if(IsTrain):
            self.PathsOfDirs=self.PathsOfDirs[0:int(len(self.PathsOfDirs)*0.9)]
        else:self.PathsOfDirs=self.PathsOfDirs[int(len(self.PathsOfDirs)*0.9):]
        self.target = [0] * len(self.PathsOfDirs)
        print(len(self.PathsOfDirs))

        Temp_dirs = os.listdir(os.path.join(self.root, "Shoushu"))
        list_index2 = list(range(len(Temp_dirs)))
        random.shuffle(list_index2)
        Temp_dirs = [Temp_dirs[index] for index in list_index2]
        if(IsTrain):
            Temp_dirs=Temp_dirs[0:int(len(Temp_dirs)*0.9)]
        else:Temp_dirs=Temp_dirs[int(len(Temp_dirs)*0.9):]
        print(len(Temp_dirs))
        Conut = 0
        for item in Temp_dirs:
            if len(os.listdir(root + "/" + "Shoushu/" + item)) >= 3:
                self.PathsOfDirs.append(os.path.join(self.root, "Shoushu", item))
                Conut += 1

        self.target.extend([1] * Conut)
        print(self.PathsOfDirs)

    def __getitem__(self, index) -> tuple:
        img_paths = os.listdir(self.PathsOfDirs[index])
        img_paths.sort(key=lambda x:x[-6])
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

        
        return img, Dict_tupianzu_to_label[self.PathsOfDirs[index][self.PathsOfDirs[index].rfind('\\')+1:]],Dict_tupianzu_to_feature[self.PathsOfDirs[index][self.PathsOfDirs[index].rfind('\\')+1:]]

    def __len__(self):
        return len(self.PathsOfDirs)


Dataset1 = DataLoader(Dataset(r"D:\ZJF\pythongrams\coding\Datas",True), batch_size=24*4, shuffle=True,drop_last=False)
Dataset2 = DataLoader(Dataset(r"D:\ZJF\pythongrams\coding\Datas",False), batch_size=24*4, shuffle=True,drop_last=False)
# for i, (imgs, labels, features) in enumerate(Dataset1):
#     pass
    # print(features.shape)
    # print(imgs.shape)
