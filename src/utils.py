import torch
import numpy as np
import os
import yaml
import random


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed=42):
    """Thiết lập random seed để kết quả tái lập được"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metrics(pred_logits, target, threshold=0.5):
    """
    Tính các chỉ số đánh giá: Dice, IoU, Accuracy, Precision, Recall.
    Input:
        pred_logits: (Batch, 1, H, W) - chưa qua sigmoid
        target: (Batch, 1, H, W) - binary mask (0, 1)
    """
    # 1. Chuyển Logits thành Binary Mask
    pred_prob = torch.sigmoid(pred_logits)
    pred_mask = (pred_prob > threshold).float()

    # 2. Flatten để tính toán
    pred_flat = pred_mask.view(-1)
    target_flat = target.view(-1)

    # 3. Tính TP, FP, TN, FN
    TP = (pred_flat * target_flat).sum()
    TN = ((1 - pred_flat) * (1 - target_flat)).sum()
    FP = (pred_flat * (1 - target_flat)).sum()
    FN = ((1 - pred_flat) * target_flat).sum()

    smooth = 1e-6  # Tránh chia cho 0

    # 4. Công thức metrics (theo bài báo)
    dice = (2 * TP + smooth) / (2 * TP + FP + FN + smooth)
    iou = (TP + smooth) / (TP + FP + FN + smooth)
    accuracy = (TP + TN + smooth) / (TP + TN + FP + FN + smooth)
    recall = (TP + smooth) / (TP + FN + smooth)
    precision = (TP + smooth) / (TP + FP + smooth)

    return {
        "dice": dice.item(),
        "iou": iou.item(),
        "accuracy": accuracy.item(),
        "recall": recall.item(),
        "precision": precision.item()
    }


class AverageMeter:
    """Class giúp tính trung bình cộng dồn trong vòng lặp"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count