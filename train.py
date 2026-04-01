import os
import json
import random
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim

from src.model import TransAttUnet
from src.dataset import TransAttUnetDataset
from src.loss import AttentionWeightedFocalTverskyLoss
from src.utils import load_config, set_seed, calculate_metrics, AverageMeter


def auto_generate_kfold(orig_split_path, kfold_split_path, k=5):
    """
    Tự động sinh file K-Fold Split nếu chưa tồn tại.
    """
    if os.path.exists(kfold_split_path):
        print(f"✅ Đã tìm thấy file K-Fold split tại: {kfold_split_path}")
        return

    print(f"⚙️ Đang tự động trộn và chia {k}-Fold từ: {orig_split_path}...")

    with open(orig_split_path, 'r') as f:
        splits = json.load(f)

    # Gom Train và Val cũ thành một rổ chung
    dev_files = splits.get('train', []) + splits.get('val', [])

    # Xáo trộn ngẫu nhiên một cách công bằng
    random.seed(42)
    random.shuffle(dev_files)

    # Két sắt: Tập Test tuyệt đối không đụng vào
    test_files = splits.get('test', [])

    fold_size = len(dev_files) // k
    kfold_splits = {'test': test_files}

    for i in range(k):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < k - 1 else len(dev_files)

        kfold_splits[f'fold_{i}'] = {
            'train': dev_files[:start_idx] + dev_files[end_idx:],
            'val': dev_files[start_idx:end_idx]
        }

    with open(kfold_split_path, 'w') as f:
        json.dump(kfold_splits, f, indent=4)
    print(f"✅ Đã tạo thành công file K-Fold: {kfold_split_path}")

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()

    # Tạo một list rỗng để trả về, tránh làm báo lỗi unpack ở hàm main()
    dummy_class_losses = []

    pbar = tqdm(loader, desc="Training", leave=False)
    for i, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device).squeeze(1).long()

        optimizer.zero_grad()

        # 1. Forward pass (Lấy cả output và attention weights)
        outputs, attn_weights = model(images)

        # 2. Tính Loss (Hỗ trợ Deep Supervision)
        if isinstance(outputs, list):
            # outputs[0] là ảnh chính
            loss_main, dice_score = criterion(outputs[0], masks, attn_weights)

            # outputs[1] và outputs[2] là nhánh phụ
            loss_ds1, _ = criterion(outputs[1], masks, attn_weights)
            loss_ds2, _ = criterion(outputs[2], masks, attn_weights)

            # Tổng hợp Loss
            loss = loss_main + 0.5 * loss_ds1 + 0.5 * loss_ds2
        else:
            loss, dice_score = criterion(outputs, masks, attn_weights)

        # 3. Backward & Optimize
        loss.backward()
        optimizer.step()

        # 4. Cập nhật log
        loss_meter.update(loss.item(), images.size(0))
        dice_meter.update(dice_score.item(), images.size(0))

        # Update thanh tiến trình
        pbar.set_postfix({'loss': f"{loss_meter.avg:.4f}", 'dice': f"{dice_meter.avg:.4f}"})

    return loss_meter.avg, dice_meter.avg, dummy_class_losses

def validate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()

    metrics_meters = {
        "dice": AverageMeter(),
        "iou": AverageMeter(),
        "accuracy": AverageMeter(),
        "recall": AverageMeter(),
        "precision": AverageMeter()
    }

    num_classes = 3  # hoặc cfg
    class_dice_meters = [AverageMeter() for _ in range(num_classes)]

    with torch.no_grad():
        for i, (images, masks) in tqdm(enumerate(loader), desc="Validating", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            masks = masks.squeeze(1).long()

            outputs, attn_weights = model(images)

            # ===== DEBUG PROBABILITY =====
            if i == 0:
                probs = torch.softmax(outputs, dim=1)
                print(f"\n[DEBUG] Max prob: {probs.max().item():.4f}")

            # ===== LOSS =====
            loss, dice_score = criterion(outputs, masks, attn_weights)
            loss_meter.update(loss.item(), images.size(0))

            # ===== METRICS =====
            scores = calculate_metrics(outputs, masks)

            for k in metrics_meters.keys():
                metrics_meters[k].update(scores[k], images.size(0))

            # ===== UPDATE CLASS DICE =====
            for cls_idx, d in enumerate(scores['dice_per_class']):
                class_dice_meters[cls_idx].update(d, images.size(0))

            # ===== DEBUG batch đầu =====
            if i == 0:
                class_dice_str = ", ".join([f"{d:.4f}" for d in scores['dice_per_class']])
                print(f"[DEBUG] Tổng Loss: {loss.item():.4f}")
                print(f"[DEBUG] Dice per class: {class_dice_str}")

    return loss_meter.avg, {
        **{k: v.avg for k, v in metrics_meters.items()},
        "dice_per_class": [m.avg for m in class_dice_meters]
    }


def save_checkpoint(state, is_best, checkpoint_dir):
    filename = os.path.join(checkpoint_dir, "last_checkpoint.pth")
    torch.save(state, filename)

    if is_best:
        best_filename = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(state['model_state_dict'], best_filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg['train']['seed'])

    os.makedirs(cfg['paths']['checkpoint_dir'], exist_ok=True)

    device = torch.device(cfg['train']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # ===== DATA =====
    train_ds = TransAttUnetDataset(
        cfg['paths']['modal_processed_data'],
        cfg['paths']['modal_split_file'],
        mode='train'
    )

    val_ds = TransAttUnetDataset(
        cfg['paths']['modal_processed_data'],
        cfg['paths']['modal_split_file'],
        mode='val'
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=cfg['data']['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['train']['batch_size'],
        shuffle=False,
        num_workers=cfg['data']['num_workers'],
        pin_memory=True
    )

    # ===== MODEL =====
    model = TransAttUnet(
        n_channels=cfg['model']['architecture']['n_channels'],
        n_classes=cfg['model']['architecture']['n_classes']
    ).to(device)

    # ===== LOSS (có weight) =====
    class_weights = cfg['train']['loss'].get('class_weights', [0.1, 0.1, 0.8])
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Đọc tham số Focal Tversky từ file config.yaml
    # criterion = FocalTverskyBoundaryLoss(
    #     alpha=cfg['train']['loss']['alpha'],
    #     beta=cfg['train']['loss']['beta'],
    #     gamma=cfg['train']['loss']['gamma'],
    #     smooth=cfg['train']['loss']['smooth'],
    #     class_weights=class_weights,
    #     boundary_weight=0.3
    # )
    criterion = AttentionWeightedFocalTverskyLoss(alpha=0.3, beta=0.7)
    # ===== OPTIMIZER =====
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg['train']['optimizer']['lr'],
        weight_decay=cfg['train']['optimizer']['weight_decay']
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg['train']['epochs'],
        eta_min=1e-6
    )

    # ===== RESUME =====
    start_epoch = 1
    best_dice = 0.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint['best_dice']

        print(f"Resumed from epoch {checkpoint['epoch']} | Best Dice: {best_dice:.4f}")

    # ===== TRAIN =====
    patience = cfg['train'].get('early_stopping', 20)
    no_improve_count = 0

    for epoch in range(start_epoch, cfg['train']['epochs'] + 1):
        print(f"\nEpoch [{epoch}]")

        train_loss, train_dice, train_class_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_metrics = validate(
            model, val_loader, criterion, device
        )

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Dice: {train_dice:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Dice: {val_metrics['dice']:.4f}")

        class_dice_str = ", ".join([f"{d:.4f}" for d in val_metrics['dice_per_class']])
        print(f"Val Dice per class: {class_dice_str}")

        # Kiểm tra có cải thiện Dice
        is_best = val_metrics['dice'] > best_dice
        if is_best:
            best_dice = val_metrics['dice']
            no_improve_count = 0
            print(f"New Best Dice: {best_dice:.4f}")
        else:
            no_improve_count += 1
            print(f"No improvement for {no_improve_count}/{patience} epochs")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_dice': best_dice
        }

        if is_best or epoch % cfg['train']['save_interval'] == 0:
            save_checkpoint(checkpoint, is_best, cfg['paths']['checkpoint_dir'])

        if no_improve_count >= patience:
            print(f"Early stopping triggered at epoch {epoch}. Best Dice: {best_dice:.4f}")
            break

    print(f"\nTraining Done | Best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()