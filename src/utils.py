import torch
import numpy as np
import os
import yaml
import random


def load_config(config_path):
    """Load YAML config"""
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


def calculate_metrics(pred_logits, target, num_classes=3):
    """
    Tính các metric: dice, iou, precision, recall, accuracy
    pred_logits: [B, C, H, W]
    target: [B, H, W] hoặc [B, 1, H, W]
    """
    preds = torch.argmax(pred_logits, dim=1)

    # Nếu target có channel singleton, loại bỏ
    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)

    smooth = 1e-6

    dice_list = []
    iou_list = []
    precision_list = []
    recall_list = []

    for cls in range(num_classes):
        pred_c = (preds == cls).float()
        target_c = (target == cls).float()

        TP = (pred_c * target_c).sum()
        FP = (pred_c * (1 - target_c)).sum()
        FN = ((1 - pred_c) * target_c).sum()

        dice = (2 * TP + smooth) / (2 * TP + FP + FN + smooth)
        iou = (TP + smooth) / (TP + FP + FN + smooth)
        precision = (TP + smooth) / (TP + FP + smooth)
        recall = (TP + smooth) / (TP + FN + smooth)

        dice_list.append(dice)
        iou_list.append(iou)
        precision_list.append(precision)
        recall_list.append(recall)

    return {
        "dice": torch.mean(torch.stack(dice_list)).item(),
        "dice_per_class": [d.item() for d in dice_list],
        "iou": torch.mean(torch.stack(iou_list)).item(),
        "precision": torch.mean(torch.stack(precision_list)).item(),
        "recall": torch.mean(torch.stack(recall_list)).item(),
        "accuracy": (preds == target).float().mean().item()
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