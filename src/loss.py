import torch
import torch.nn as nn
import torch.nn.functional as F


class TransAttLoss(nn.Module):
    """
    Hàm Loss kết hợp theo công thức (9) trong bài báo:
    Loss = alpha * BCE + beta * Dice
    """

    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super(TransAttLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        logits: Output từ model (chưa qua Sigmoid), shape (B, 1, H, W)
        targets: Ground truth mask, shape (B, 1, H, W)
        """
        # 1. Binary Cross Entropy Loss
        bce_loss = self.bce(logits, targets)

        # 2. Dice Loss
        # Dice = 1 - (2 * Intersection + epsilon) / (Union + epsilon)
        pred = torch.sigmoid(logits)  # Chuyển logits thành xác suất [0, 1]

        # Flatten để tính toán trên toàn bộ batch
        pred_flat = pred.view(-1)
        targets_flat = targets.view(-1)

        intersection = (pred_flat * targets_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (pred_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1 - dice_score

        # 3. Tổng hợp
        total_loss = (self.alpha * bce_loss) + (self.beta * dice_loss)

        return total_loss, dice_score  # Trả về cả Dice Score để log