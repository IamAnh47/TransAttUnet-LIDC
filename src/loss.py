import torch
import torch.nn as nn
import torch.nn.functional as F

class TransAttLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, class_weights=None, smooth=1e-6):
        super().__init__()
        self.alpha = alpha  # CE weight
        self.beta = beta    # Dice weight
        self.smooth = smooth

        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

    def forward(self, outputs, targets, return_class_losses=False):
        """
        outputs: [B, C, H, W]
        targets: [B, H, W] long
        """
        B, C, H, W = outputs.shape
        device = outputs.device

        targets_onehot = F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2).float()  # [B, C, H, W]

        # ----- Cross Entropy -----
        ce_loss = F.cross_entropy(outputs, targets, weight=self.class_weights.to(device) if self.class_weights is not None else None, reduction='none')  # [B, H, W]
        ce_loss_per_class = []
        for c in range(C):
            mask_c = (targets == c).float()
            loss_c = (ce_loss * mask_c).sum() / (mask_c.sum() + 1e-6)
            ce_loss_per_class.append(loss_c.item())

        ce_loss = ce_loss.mean()

        # ----- Dice Loss -----
        probs = F.softmax(outputs, dim=1)
        dice_loss_per_class = []
        dice_total = 0.0
        for c in range(C):
            p = probs[:, c, :, :]
            t = targets_onehot[:, c, :, :]
            intersection = (p * t).sum()
            union = p.sum() + t.sum()
            dice_c = 1 - (2 * intersection + self.smooth) / (union + self.smooth)
            dice_loss_per_class.append(dice_c.item())
            dice_total += dice_c

        dice_loss = dice_total / C

        # ----- Total Loss -----
        loss = self.alpha * ce_loss + self.beta * dice_loss

        if return_class_losses:
            class_losses = [self.alpha * ce + self.beta * dice for ce, dice in zip(ce_loss_per_class, dice_loss_per_class)]
            return loss, 1 - dice_loss, class_losses  # Dice tổng, loss tổng, loss từng class
        else:
            return loss, 1 - dice_loss  # Dice tổng