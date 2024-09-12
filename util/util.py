# By Yuxiang Sun, Dec. 4, 2020
# Email: sun.yuxiang@outlook.com

import numpy as np 
from PIL import Image 
import cv2
import os
 
# 0:unlabeled, 1:gas
def get_palette():
    unlabelled = [0,0,0]
    gas        = [64,0,128]
    palette    = np.array([unlabelled,gas])
    return palette

def visualize(image_name, predictions):
    palette = get_palette()
    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy()
        img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cid in range(0, len(palette)): # fix the mistake from the MFNet code on Dec.27, 2019
            img[pred == cid] = palette[cid]
        img = Image.fromarray(np.uint8(img))
        img.save('./results2/' + image_name[i] + '.png')

def show(image_name, predictions, weight_name, origin):
    palette = get_palette()
    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy()
        img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cid in range(0, len(palette)): # fix the mistake from the MFNet code on Dec.27, 2019
            img[pred == cid] = palette[cid]
        thermal = origin.transpose((2,0,1))[3]
        thermal = cv2.cvtColor(thermal,cv2.COLOR_GRAY2BGR)
        img = cv2.addWeighted(thermal,0.5,img,0.5,0)
        img = np.uint8(img)
        cv2.imwrite('result.png',img)
        # cv2.imshow('img',img)
        cv2.waitKey(0)

def visualize_gt(label_path, image_name, predictions, weight_name):
    label_path = label_path
    palette = get_palette()
    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy()
        gt = cv2.imread(os.path.join(label_path,image_name[i]+".png"))
        gt = gt * 255
        # Convert the expanded image to uint8 data type
        gt = gt.astype(np.uint8)
        img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cid in range(0, len(palette)): # fix the mistake from the MFNet code on Dec.27, 2019
            img[pred == cid] = palette[cid]
        output = cv2.addWeighted(img, 0.5, gt, 0.5 ,0)
        img = Image.fromarray(np.uint8(output))
        img.save('./results3/' + image_name[i] + '.png')

# def visualize(label_path, image_name, predictions, weight_name):
#     label_path = label_path
#     palette = get_palette()
#     for (i, pred) in enumerate(predictions):
#         pred = predictions[i].cpu().numpy()
#         gt = cv2.imread(os.path.join(label_path,image_name[i]+".png"))
#         gt = gt * 255
#         # Convert the expanded image to uint8 data type
#         gt = gt.astype(np.uint8)
#         img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
#         for cid in range(0, len(palette)): # fix the mistake from the MFNet code on Dec.27, 2019
#             img[pred == cid] = palette[cid]
#         output = cv2.addWeighted(img, 0.5, gt, 0.5 ,0)
#         img = Image.fromarray(np.uint8(output))
#         img.save('./results/' + image_name[i] + '.png')

def compute_results(conf_total):
    n_class =  conf_total.shape[0]
    consider_unlabeled = True  # must consider the unlabeled, please set it to True
    if consider_unlabeled is True:
        start_index = 0
    else:
        start_index = 1
    precision_per_class = np.zeros(n_class)
    recall_per_class = np.zeros(n_class)
    iou_per_class = np.zeros(n_class)
    for cid in range(start_index, n_class): # cid: class id

        if conf_total[start_index:, cid].sum() == 0:
            precision_per_class[cid] =  np.nan
        else:
            precision_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[start_index:, cid].sum()) # precision = TP/TP+FP
        if conf_total[cid, start_index:].sum() == 0:
            recall_per_class[cid] = np.nan
        else:
            recall_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[cid, start_index:].sum()) # recall = TP/TP+FN
        if (conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid]) == 0:
            iou_per_class[cid] = np.nan
        else:
            iou_per_class[cid] = float(conf_total[cid, cid]) / float((conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid])) # IoU = TP/TP+FP+FN

    return precision_per_class, recall_per_class, iou_per_class

def compute_results2(conf_total, beta=1.0):
    n_class = conf_total.shape[0]  # 混淆矩阵中类别的数量
    consider_unlabeled = True  # 是否考虑未标记类别，请设置为True
    if consider_unlabeled is True:
        start_index = 0
    else:
        start_index = 1
    
    precision_per_class = np.zeros(n_class)  # 存储每个类别的精确率
    recall_per_class = np.zeros(n_class)  # 存储每个类别的召回率
    iou_per_class = np.zeros(n_class)  # 存储每个类别的交并比
    f1_per_class = np.zeros(n_class)  # 存储每个类别的F1分数
    f2_per_class = np.zeros(n_class)  # 存储每个类别的F2分数
    f0_5_per_class = np.zeros(n_class)  # 存储每个类别的F0.5分数
    
    for cid in range(start_index, n_class):
        # 计算当前类别的精确率
        if conf_total[start_index:, cid].sum() == 0:
            precision_per_class[cid] = np.nan
        else:
            precision_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[start_index:, cid].sum())  # 精确率 = TP / (TP + FP)
        
        # 计算当前类别的召回率
        if conf_total[cid, start_index:].sum() == 0:
            recall_per_class[cid] = np.nan
        else:
            recall_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[cid, start_index:].sum())  # 召回率 = TP / (TP + FN)
        
        # 计算当前类别的交并比
        if (conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid]) == 0:
            iou_per_class[cid] = np.nan
        else:
            iou_per_class[cid] = float(conf_total[cid, cid]) / float((conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid])) # IoU = TP/TP+FP+FN

        # 计算F1分数
        if precision_per_class[cid] + recall_per_class[cid] == 0:
            f1_per_class[cid] = np.nan
        else:
            f1_per_class[cid] = (2 * precision_per_class[cid] * recall_per_class[cid]) / (precision_per_class[cid] + recall_per_class[cid])
        
        # 计算F2分数
        beta2 = (2*beta) ** 2
        if (beta2 * precision_per_class[cid] + recall_per_class[cid]) == 0:
            f2_per_class[cid] = np.nan
        else:
            f2_per_class[cid] = (1 + beta2) * precision_per_class[cid] * recall_per_class[cid] / (beta2 * precision_per_class[cid] + recall_per_class[cid])
        
        # 计算F0.5分数
        beta0_5 = (beta / 2) **2
        if (precision_per_class[cid] + beta0_5 * recall_per_class[cid]) == 0:
            f0_5_per_class[cid] = np.nan
        else:
            f0_5_per_class[cid] = (1 + beta0_5) * precision_per_class[cid] * recall_per_class[cid] / (precision_per_class[cid] + beta0_5 * recall_per_class[cid])
    
    return precision_per_class, recall_per_class, iou_per_class, f1_per_class, f2_per_class, f0_5_per_class
