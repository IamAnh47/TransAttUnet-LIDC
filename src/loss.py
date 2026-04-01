import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionWeightedFocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets, attn_weights):
        B, C, H, W = logits.shape
        device = logits.device

        # --- 1. XỬ LÝ BẢN ĐỒ ATTENTION ---
        attn_mean = attn_weights.mean(dim=1)
        attn_spatial = attn_mean.sum(dim=1)

        H_bot = int(attn_spatial.shape[1] ** 0.5)
        attn_spatial = attn_spatial.view(B, 1, H_bot, H_bot)

        # Phóng to
        attn_upsampled = F.interpolate(attn_spatial, size=(H, W), mode='bilinear', align_corners=False)

        # Chuẩn hóa 0-1
        attn_min = attn_upsampled.view(B, -1).min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        attn_max = attn_upsampled.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        attn_norm = (attn_upsampled - attn_min) / (attn_max - attn_min + 1e-8)

        # A_i = 1 + attention (Khuếch đại phạt ở những vùng đáng chú ý)
        A_i = 1.0 + attn_norm

        # --- 2. TÍNH FOCAL TVERSKY CÓ TRỌNG SỐ ---
        targets_onehot = F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2).float()
        probs = torch.softmax(logits, dim=1)

        ft_losses = []
        dice_scores = []

        # Chạy từ class 1 (bỏ qua background)
        for c in range(1, C):
            p = probs[:, c, :, :]
            t = targets_onehot[:, c, :, :]
            a = A_i[:, 0, :, :] # Ma trận trọng số

            # Nhân trọng số Attention thẳng vào TP, FP, FN
            TP = (a * p * t).sum()
            FP = (a * p * (1 - t)).sum()
            FN = (a * (1 - p) * t).sum()

            # Tversky có Attention
            tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
            ft_loss = (1 - tversky) ** self.gamma
            ft_losses.append(ft_loss)

            # Tính Dice thuần túy để in log
            TP_raw = (p * t).sum()
            FP_raw = (p * (1 - t)).sum()
            FN_raw = ((1 - p) * t).sum()
            dice_c = (2 * TP_raw + self.smooth) / (2 * TP_raw + FP_raw + FN_raw + self.smooth)
            dice_scores.append(dice_c.item())

        mean_loss = torch.stack(ft_losses).mean()
        mean_dice = sum(dice_scores) / len(dice_scores)

        return mean_loss, torch.tensor(mean_dice, device=device)

class BoundaryLoss(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        # Dùng kernel 5x5 để tạo ra một đường viền có độ dày khoảng 2-3 pixel
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, inputs, targets):
        # Dùng Max Pooling để làm phép Dilation (Phình to) và Erosion (Co lại)
        # Dilation: Viền ngoài
        dilation_in = F.max_pool2d(inputs, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        # Erosion: Lõi bên trong
        erosion_in = -F.max_pool2d(-inputs, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        # Đường viền = Phình to - Lõi
        boundary_in = dilation_in - erosion_in

        # Làm tương tự với Ground Truth
        dilation_tar = F.max_pool2d(targets, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        erosion_tar = -F.max_pool2d(-targets, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        boundary_tar = dilation_tar - erosion_tar

        # Ép đường viền dự đoán phải giống hệt đường viền thật bằng MSE
        return F.mse_loss(boundary_in, boundary_tar)


class FocalTverskyBoundaryLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1e-6, class_weights=None, boundary_weight=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.boundary_weight = boundary_weight  # Trọng số cho Boundary Loss (thường để 0.2 - 0.5)

        self.boundary_loss_fn = BoundaryLoss(kernel_size=5)

        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

    def forward(self, outputs, targets, return_class_losses=False):
        B, C, H, W = outputs.shape
        device = outputs.device

        probs = F.softmax(outputs, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2).float()

        if self.class_weights is not None:
            weights = self.class_weights.to(device)
        else:
            weights = torch.ones(C).to(device)

        ft_losses = []
        dice_scores = []

        # 1. Tính Boundary Loss (chỉ tính cho class 1 và 2, bỏ qua background class 0 để tối ưu)
        b_loss = self.boundary_loss_fn(probs[:, 1:, :, :], targets_onehot[:, 1:, :, :])

        # 2. Tính Focal Tversky Loss như cũ
        for c in range(C):
            p = probs[:, c, :, :]
            t = targets_onehot[:, c, :, :]

            TP = (p * t).sum()
            FP = (p * (1 - t)).sum()
            FN = ((1 - p) * t).sum()

            tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
            ft_loss = (1 - tversky) ** self.gamma

            ft_losses.append(ft_loss * weights[c])

            # Tính Dice để in log
            dice_c = (2 * TP + self.smooth) / (2 * TP + FP + FN + self.smooth)
            dice_scores.append(dice_c.item())

        total_ft_loss = torch.stack(ft_losses).mean()
        mean_dice = sum(dice_scores) / C

        # 3. TỔNG HỢP LOSS
        total_loss = total_ft_loss + self.boundary_weight * b_loss

        if return_class_losses:
            class_losses_vals = [l.item() for l in ft_losses]
            return total_loss, torch.tensor(mean_dice, device=device), class_losses_vals
        else:
            return total_loss, torch.tensor(mean_dice, device=device)