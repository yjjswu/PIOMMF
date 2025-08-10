import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize
from itertools import cycle, product
import numpy as np

import timm.models.rdnet
from DataOfClass import Dataset1,Dataset2
# [0.01, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,]).cuda()
from StarDN_Frame import *
import numpy as np
from funcs import *
import random
from numpy.core.multiarray import _reconstruct
# from
# def focal_loss(pred: torch.Tensor, target: torch.Tensor, alpha=0.25, gamma=2.0):
#     # target_temp = target.long().cpu().detach().numpy()
#     #
#     # target_temp = target_temp[0,0]
#     # # print(np.max(target_temp))
#     # plt.imshow((target_temp / 6) * 255)
#     # plt.show()
#     target = target[:, 0, :, :].long()
#
#     bce_loss = Cross1(pred, target)
#     pt = torch.exp(-bce_loss)
#     focal_loss = alpha * (1 - pt) ** gamma * bce_loss
#     return focal_loss.mean()

def giou_loss(preds, targets):
    """
    Compute GIoU loss between predicted and target boxes.

    Args:
    preds (Tensor): Predicted bounding boxes, shape (N, 4).
    targets (Tensor): Target bounding boxes, shape (N, 4).

    Returns:
    Tensor: GIoU loss.
    """
    # Predicted boxes
    pred_x1, pred_y1, pred_x2, pred_y2 = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]

    # Target boxes
    target_x1, target_y1, target_x2, target_y2 = targets[:, 0], targets[:, 1], targets[:, 2], targets[:, 3]

    # Intersection coordinates
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)

    # Intersection area
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Areas of the predicted and target boxes
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)

    # Union area
    union_area = pred_area + target_area - inter_area

    # IoU
    iou = inter_area / union_area

    # Smallest enclosing box coordinates
    enc_x1 = torch.min(pred_x1, target_x1)
    enc_y1 = torch.min(pred_y1, target_y1)
    enc_x2 = torch.max(pred_x2, target_x2)
    enc_y2 = torch.max(pred_y2, target_y2)

    # Area of the smallest enclosing box
    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)

    # GIoU
    giou = iou - (enc_area - union_area) / enc_area

    # GIoU loss
    loss = 1 - giou
    return loss.mean()


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
# 新增导入
from torch.cuda.amp import autocast, GradScaler

def train_model(model, dataloader, dataloaderx, num_classes, num_epochs=400):
    Cross1 = nn.CrossEntropyLoss(weight=torch.Tensor([1,1.6,1.6,1.6]).cuda())
    optimizer = optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.12)
    scher1 = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.4)
    
    # 初始化GradScaler
    scaler = GradScaler()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, targets, features) in enumerate(dataloader):
            optimizer.zero_grad()
            images = images.cuda()
            targets = targets.cuda()
            # features = features.cuda().type(torch.float32)
            
            rot_degree = np.pi*(random.randint(-15,15)/180.0)
            images_rot = rot_img(images, rot_degree)
            
            # 使用自动混合精度
            # with autocast():
            outputs1 = model(images_rot)
            loss1 = Cross1(outputs1, targets)
            # outputs1 = model(images,features)
            # loss1 = Cross1(outputs1, targets)
            # 反向传播和优化步骤
            # scaler.scale(loss1).backward()
            # scaler.step(optimizer)
            # scaler.update()
            loss1.backward()
            optimizer.step()
            
            running_loss += loss1.item()
            
            if (i % 2 == 0):
                print(f"temp_loss: {loss1.item()}")
                torch.save(model, "./weights/RDNet_Ori.pt")
                with open('./run/RDNet_Ori.txt', 'a') as file_loss:
                    file_loss.write(' ' + str(loss1.item()))
        
        scher1.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}')
        
        if epoch % 5 == 0:
            model.eval()
            # 验证集评估
            
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for img, tgt, feat in dataloader:
                  img = img.cuda()
                  tgt = tgt.cuda()
                  # feat = feat.cuda().type(torch.float32)
                  
                  # 验证时使用混合精度
                  # with autocast():
                  outputs = model(img)
                  # outputs = outputs.float()  # 转换为FP32用于后续计算
                  
                  _, preds = torch.max(outputs, 1)
                  all_preds.extend(preds.cpu().numpy())
                  all_targets.extend(tgt.cpu().numpy())
            
            # 计算并打印指标
            accuracy = accuracy_score(all_targets, all_preds)
            recall = recall_score(all_targets, all_preds, average='micro')
            precision = precision_score(all_targets, all_preds, average='micro')
            f1 = f1_score(all_targets, all_preds, average='micro')
            
            print(f'Accuracy: {accuracy:.4f}')
            print(f'Recall: {recall:.4f}')
            print(f'Precision: {precision:.4f}')
            print(f'F1 Score: {f1:.4f}')
            print(classification_report(all_targets, all_preds, 
                                  target_names=['Class 0', 'Class 1','Class 2','Class 3']))


            all_preds = []
            all_targets = []
            with torch.no_grad():  # 禁用梯度计算
                for images, targets,features in dataloaderx:
                    for i in range(5):
                          images = images.cuda()+torch.randn_like(images).cuda()*0.12
                          targets = targets.cuda()
                          # features = features.cuda().type(torch.float32)
                          outputs = model(images)
                          # 假设输出是概率分布，取最大值的索引作为预测类别
                          _, preds = torch.max(outputs, 1)
                          # 将预测结果和真实标签收集起来
                          all_preds.extend(preds.cpu().numpy())
                          all_targets.extend(targets.cpu().numpy())
            # 计算各类指标
            accuracy = accuracy_score(all_targets, all_preds)
            recall = recall_score(all_targets, all_preds, average='micro')  # 二分类情况
            precision = precision_score(all_targets, all_preds, average='micro')
            f1 = f1_score(all_targets, all_preds, average='micro')
            
            print(f'Accuracy: {accuracy:.4f}')
            print(f'Recall: {recall:.4f}')
            print(f'Precision: {precision:.4f}')
            print(f'F1 Score: {f1:.4f}')
            
            # 如果需要更详细的分类报告，可以使用sklearn的classification_report
            # from sklearn.metrics import classification_report
            print("\nClassification Report:")
            print(classification_report(all_targets, all_preds, target_names=['Class 0', 'Class 1','Class 2','Class 3']))

def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # Update the indices
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[0] + box[2], boxes[:, 0] + boxes[:, 2])
    ymax = np.minimum(box[1] + box[3], boxes[:, 1] + boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = box[2] * box[3]
    boxes_area = boxes[:, 2] * boxes[:, 3]
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def Test_Model(Modelx: nn.Module, dataloaderx: DataLoader):
    Modelx = Modelx.cpu()
    # 假设类别标签
    Change_Dict3 = {1: 'missing_hole', 2: 'mouse_bite', 3: 'open_circuit', 4: 'short', 5: 'spur', 6: 'spurious_copper'}
    Change_Dict2 = {1: 'open', 2: 'short', 3: 'mousebite', 4: 'spur', 5: 'pin_hole', 6: 'spurious _copper'}
    for i, (images, targets) in enumerate(dataloaderx):
        Output = Modelx(images).detach().numpy()

        Temp_heatmap = np.mean(Output[0, :7], axis=0)
        plt.figure(figsize=(10, 10))
        plt.imshow(Temp_heatmap / 7 * 255)
        plt.show()

        A = Output[0, :7, :, :].argmax(axis=0)
        print(np.max(A / 7.0 * 255.0))
        plt.figure(figsize=(10, 10))
        plt.imshow(A / 7.0 * 255.0)
        plt.show()

        B = targets[0, 0, :, :].detach().numpy()
        plt.figure(figsize=(10, 10))
        plt.imshow(B * 255)
        plt.show()

        Img_Ori = images[0].permute(1, 2, 0).detach().numpy()
        plt.figure(figsize=(10, 10))
        plt.imshow(Img_Ori)
        plt.show()

        non_one_indices = np.argwhere(A != 0)
        print("A 中不为 1 的元素的坐标索引:")
        print(non_one_indices)
        print(f"共有 {len(non_one_indices)} 个不为 1 的元素")

        non_one_indices2 = np.argwhere(B != 0)

        plt.figure(figsize=(10, 10))
        plt.imshow(Img_Ori)
        B2 = targets[0, 1:]
        for item in non_one_indices2:
            Temp1 = B2[:, item[0], item[1]]

            x0 = ((item[1] + Temp1[0] * 56) / 56 - Temp1[2] / 2) * 448
            y0 = ((item[0] + Temp1[1] * 56) / 56 - Temp1[3] / 2) * 448
            w0 = Temp1[2] * 448
            h0 = Temp1[3] * 448
            Class_Name = B[item[0], item[1]]
            # print(x0, y0, w0, h0)
            rect = plt.Rectangle((x0, y0), w0, h0, linewidth=1, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)
            plt.text(x0, y0, f"{Change_Dict2[Class_Name]}", color='blue', fontsize=6,
                     bbox=dict(facecolor='white', alpha=0.3))
        plt.xlim(0, Img_Ori.shape[1])
        plt.ylim(Img_Ori.shape[0], 0)
        plt.show()

        plt.figure(figsize=(10, 10))
        plt.imshow(Img_Ori)
        A2 = Output[0, 7:, :, :]

        # Prepare boxes and scores for NMS
        boxes = []
        scores = []
        Classes = []
        for item in non_one_indices:
            Temp1 = A2[:, item[0], item[1]]
            x0 = ((item[1] + Temp1[0] * 56) / 56 - Temp1[2] / 2) * 448
            y0 = ((item[0] + Temp1[1] * 56) / 56 - Temp1[3] / 2) * 448
            w0 = Temp1[2] * 448
            h0 = Temp1[3] * 448
            Class_Name = A[item[0], item[1]]
            Classes.append(Class_Name)
            boxes.append([x0, y0, w0, h0])
            scores.append(np.max(Output[0, :7, item[0], item[1]]))  # Use the highest class probability as the score

        boxes = np.array(boxes)
        scores = np.array(scores)

        # Apply NMS
        # keep_indices = nms(boxes, scores, iou_threshold=0.8)
        # print("---------------",keep_indices)
        # Draw only the boxes that survived NMS
        for idx in range(len(boxes)):
            x0, y0, w0, h0 = boxes[idx]
            rect = plt.Rectangle((x0, y0), w0, h0, linewidth=0.6, edgecolor='r', facecolor='none')
            plt.text(x0, y0, f"{Change_Dict2[Classes[idx]]}", color='blue', fontsize=6,
                     bbox=dict(facecolor='white', alpha=0.3))
            plt.gca().add_patch(rect)

        plt.xlim(0, Img_Ori.shape[1])
        plt.ylim(Img_Ori.shape[0], 0)
        plt.show()


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection

    iou = intersection / union if union > 0 else 0
    return iou


def Test_Model_MIOU(Modelx: nn.Module, dataloaderx: DataLoader):
    Modelx = Modelx.cpu()
    total_iou = 0
    total_boxes = 0

    for i, (images, targets) in enumerate(dataloaderx):
        Output = Modelx(images).detach().numpy()
        batch_size = Output.shape[0]

        for batch_idx in range(batch_size):
            A = Output[batch_idx, :7, :, :].argmax(axis=0)
            B = targets[batch_idx, 0, :, :].detach().numpy()

            non_one_indices = np.argwhere(A != 0)
            non_one_indices2 = np.argwhere(B != 0)

            B2 = targets[batch_idx, 1:]
            gt_boxes = []
            for item in non_one_indices2:
                Temp1 = B2[:, item[0], item[1]]
                x0 = ((item[1] + Temp1[0] * 56) / 56 - Temp1[2] / 2) * 448
                y0 = ((item[0] + Temp1[1] * 56) / 56 - Temp1[3] / 2) * 448
                w0 = Temp1[2] * 448
                h0 = Temp1[3] * 448
                gt_boxes.append([x0, y0, w0, h0])

            A2 = Output[batch_idx, 7:, :, :]
            pred_boxes = []
            for item in non_one_indices:
                Temp1 = A2[:, item[0], item[1]]
                x0 = ((item[1] + Temp1[0] * 56) / 56 - Temp1[2] / 2) * 448
                y0 = ((item[0] + Temp1[1] * 56) / 56 - Temp1[3] / 2) * 448
                w0 = Temp1[2] * 448
                h0 = Temp1[3] * 448
                pred_boxes.append([x0, y0, w0, h0])

            for pred_box in pred_boxes:
                ious = [compute_iou(pred_box, gt_box) for gt_box in gt_boxes]
                max_iou = max(ious) if ious else 0
                total_iou += max_iou
                total_boxes += 1

        print(f"Processed batch {i + 1}, current mean IoU: {total_iou / total_boxes if total_boxes > 0 else 0}")

    # 计算整个数据集的平均IoU（mIoU）
    miou = total_iou / total_boxes if total_boxes > 0 else 0
    print(f"Final Mean IoU: {miou}")
    return miou


def Test_Model_Metrics(Modelx: nn.Module, dataloaderx: DataLoader, iou_threshold=0.5):
    Modelx = Modelx.cpu()
    total_iou = 0
    total_boxes = 0
    correct_predictions = 0
    all_precisions = []
    all_recalls = []
    for i, (images, targets) in enumerate(dataloaderx):
        Output = Modelx(images).detach().numpy()
        batch_size = Output.shape[0]

        for batch_idx in range(batch_size):
            A = Output[batch_idx, :7, :, :].argmax(axis=0)
            B = targets[batch_idx, 0, :, :].detach().numpy()

            non_one_indices = np.argwhere(A != 0)
            non_one_indices2 = np.argwhere(B != 0)

            B2 = targets[batch_idx, 1:]
            gt_boxes = []
            for item in non_one_indices2:
                Temp1 = B2[:, item[0], item[1]]
                x0 = ((item[1] + Temp1[0] * 56) / 56 - Temp1[2] / 2) * 448
                y0 = ((item[0] + Temp1[1] * 56) / 56 - Temp1[3] / 2) * 448
                w0 = Temp1[2] * 448
                h0 = Temp1[3] * 448
                gt_boxes.append([x0, y0, w0, h0])

            A2 = Output[batch_idx, 7:, :, :]
            pred_boxes = []
            for item in non_one_indices:
                Temp1 = A2[:, item[0], item[1]]
                x0 = ((item[1] + Temp1[0] * 56) / 56 - Temp1[2] / 2) * 448
                y0 = ((item[0] + Temp1[1] * 56) / 56 - Temp1[3] / 2) * 448
                w0 = Temp1[2] * 448
                h0 = Temp1[3] * 448
                pred_boxes.append([x0, y0, w0, h0])

            for pred_box in pred_boxes:
                ious = [compute_iou(pred_box, gt_box) for gt_box in gt_boxes]
                max_iou = max(ious) if ious else 0
                total_iou += max_iou
                total_boxes += 1
                if max_iou >= iou_threshold:
                    correct_predictions += 1

                # Calculate precision and recall
                true_positives = sum(iou >= iou_threshold for iou in ious)
                if len(pred_boxes) > 0:
                    precision = true_positives / len(pred_boxes)
                else:
                    precision = 0

                recall = true_positives / len(gt_boxes) if len(gt_boxes) > 0 else 0

                all_precisions.append(precision)
                all_recalls.append(recall)
        print(f"Processed batch {i + 1}, current mean IoU: {total_iou / total_boxes if total_boxes > 0 else 0}")
    # Calculate metrics
    miou = total_iou / total_boxes if total_boxes > 0 else 0
    accuracy = correct_predictions / total_boxes if total_boxes > 0 else 0
    map_score = np.mean(all_precisions)
    print(f"Final Mean IoU: {miou}")
    print(f"Accuracy: {accuracy}")
    print(f"Recall:{recall}")
    print(f"Mean Average Precision: {map_score}")
    return miou, accuracy, map_score


def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    绘制混淆矩阵，并在对应位置标记数字。
    :param cm: 混淆矩阵
    :param classes: 类别名称列表
    :param title: 图像标题
    :param cmap: 颜色映射
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def evaluate_model(model, dataloader, num_classes):
    Change_Dict2 = {0: 'background', 1: 'open', 2: 'short', 3: 'mousebite', 4: 'spur', 5: 'pin_hole',
                    6: 'spurious _copper'}
    Change_Dict3 = {0: 'background', 1: 'missing_hole', 2: 'mouse_bite', 3: 'open_circuit', 4: 'short', 5: 'spur',
                    6: 'spurious_copper'}
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, targets in dataloader:
            outputs = model(images.cuda())
            _, preds = torch.max(outputs[:, :num_classes], 1)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(targets[:, 0, :, :].cpu().numpy().flatten())
    print(len(all_preds))
    print(len(all_labels))
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    cm[0][0] = -1
    print("混淆矩阵：")
    print(cm)

    # 绘制混淆矩阵
    classes = [str(i) for i in list(Change_Dict3.values())]  # 类别名称列表
    plot_confusion_matrix(cm, classes)
    plt.show()

    # 计算ROC曲线和AUC
    labels = label_binarize(all_labels, classes=np.arange(num_classes))
    preds = label_binarize(all_preds, classes=np.arange(num_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 绘制ROC曲线
    plt.figure()
    colors = cycle(['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # 计算PR曲线和AP
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(labels[:, i], preds[:, i])
        average_precision[i] = average_precision_score(labels[:, i], preds[:, i])

    # 绘制PR曲线
    plt.figure()
    colors = cycle(['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label='PR curve of class {0} (AP = {1:0.2f})'.format(i, average_precision[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower left")
    plt.show()


# 假设你已经加载了模型和数据集
num_classes = 4  # 根据你的数据集调整类别数
# model = torch.load('./weights/DeVoc_1226.pt').cuda()
dataloader,DataLoader2 = Dataset1,Dataset2  # 你的数据加载器
# evaluate_model(model, dataloader, num_classes)

# Test_Model(torch.load('./weights/DeVoc_DeepPCB_1_3.pt').eval(), dataloaderx=dataloader)
# get_cam()
# x1 = Test_Model_Metrics(torch.load('./weights/DeVoc_DeepPCB_1_3.pt').eval(), dataloaderx=dataloader)
# print(x1)
# 初始化模型并开始训练
# num_classes = 7  # 根据你的数据集调整类别数

# with torch.serialization.safe_globals([_reconstruct]):
# modelx=torch.load('./weights/RDNet_Ori.pt',weights_only=False).cuda()
# torch.save(modelx.state_dict(),"./weights/RDNet_1.pth")
# model = torch.load('./weights/RDNet.pt',weights_only=False).cuda()
    # model = torch.load('').cuda()
# modelx = DetectionModel(num_classes=num_classes).cuda()
# modelx=torch.load("./weights/shivit_ori.pt")
# del A['backbone.backbone.head.fc.weight']
# del A['backbone.backbone.head.fc.bias']
# model.load_state_dict(A,strict=False)
# from MedVIT import *
# modelx=MedViT_small(num_classes=num_classes).cuda()
modelx=timm.models.rdnet.rdnet_small(num_classes=num_classes).cuda()
train_model(modelx, dataloader, DataLoader2, num_classes, num_epochs=600)