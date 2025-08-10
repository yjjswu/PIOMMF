import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
from DataOfClass import Dataset1,Dataset2
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

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradient = None
        self.activation = None
        
        # Register hooks
        def forward_hook(module, input, output):
            self.activation = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradient = grad_output[0].detach()
        
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_image, input_feature, target_class=None):
        # Forward pass
        output = self.model(input_image, input_feature)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Target for backprop
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        
        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Generate CAM
        pooled_gradients = torch.mean(self.gradient, dim=[2, 3])
        for i in range(pooled_gradients.shape[1]):
            self.activation[:, i, :, :] *= pooled_gradients[0, i]
        
        heatmap = torch.mean(self.activation, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap = heatmap.cpu().numpy()
        
        # Normalize heatmap
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        return heatmap

def visualize_cam(image, heatmap, save_path=None):
    # Convert image to RGB if it's in BGR
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image[:,:,1]=image[:,:,0]
    image[:,:,2]=image[:,:,0]
    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert heatmap to uint8
    heatmap = np.uint8(255 * heatmap)
    
    # Check if heatmap has more than one channel, and convert to grayscale if necessary
    if len(heatmap.shape) > 2:
        heatmap = np.mean(heatmap, axis=2)
    
    # Convert heatmap to CV_8UC1 format
    heatmap = cv2.cvtColor(heatmap.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Combine original image with heatmap
    superimposed_img = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)
    
    # Display results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_colored)
    plt.title('Heatmap')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_img)
    plt.title('Grad-CAM')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def apply_gradcam(model, dataloader, num_samples=5):
    model.eval()
    
    # 获取目标层
    target_layer = model.backbone.stages[1]  # 根据你的模型结构调整
    # target_layer = model.backbone.stem
    grad_cam = GradCAM(model, target_layer)
    
    for i, (images, targets, features) in enumerate(dataloader):
        if i >= num_samples:
            break
            
        images = images.cuda()
        features = features.cuda().type(torch.float32)
        
        # 生成CAM
        image_np = images[0].cpu().numpy().transpose(1, 2, 0)
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        image_np = np.uint8(255 * image_np)
        
        heatmap = grad_cam.generate_cam(images, features)
        
        # 可视化
        save_path = f'gradcam_sample_{i}.png'
        visualize_cam(image_np, heatmap, save_path)

# Usage example:
if __name__ == "__main__":
    model = torch.load('./weights/TestHyper_0419.pt')
    model.cuda()
    dataloader = Dataset2  # Your dataloader
    
    apply_gradcam(model, dataloader)