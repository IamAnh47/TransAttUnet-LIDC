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
from src.loss import MultiTaskLoss, AttentionWeightedFocalTverskyLoss
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

    if k == 1:
        print("⚠️ Chế độ k=1: Bỏ qua chia K-Fold, sử dụng lại đúng tập Train/Val gốc.")
        kfold_splits = {
            'test': splits.get('test', []),
            'fold_0': {
                'train': splits.get('train', []),
                'val': splits.get('val', [])
            }
        }
        with open(kfold_split_path, 'w') as f:
            json.dump(kfold_splits, f, indent=4)
        print(f"✅ Đã tạo file config 1-Fold tại: {kfold_split_path}")
        return

    if os.path.exists(kfold_split_path):
        # Kiểm tra dọn dẹp file hỏng
        with open(kfold_split_path, 'r') as f:
            check_splits = json.load(f)
            if not check_splits.get('fold_0', {}).get('train'):
                print("⚠️ Phát hiện file K-Fold cũ bị rỗng tập Train. Đang xóa để tạo lại...")
                os.remove(kfold_split_path)
            else:
                print(f"✅ Đã tìm thấy file K-Fold split hợp lệ tại: {kfold_split_path}")
                return

    print(f"⚙️ Đang tự động trộn và chia {k}-Fold từ: {orig_split_path}...")

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

        # 1. Forward pass (Bây giờ lấy cả output của nhánh tái tạo)
        outputs, recon_outputs, attn_weights = model(images)

        # 2. Tính Loss (Hỗ trợ Deep Supervision và Multi-Task)
        if isinstance(outputs, list):
            # Tính Loss nhánh chính (có DS)
            loss_main, dice_score = criterion.seg_loss_fn(outputs[0], masks, attn_weights)
            loss_ds1, _ = criterion.seg_loss_fn(outputs[1], masks, attn_weights)
            loss_ds2, _ = criterion.seg_loss_fn(outputs[2], masks, attn_weights)
            seg_loss_total = loss_main + 0.5 * loss_ds1 + 0.5 * loss_ds2

            # Tính Loss nhánh tái tạo (Ép ảnh đầu ra phải giống ảnh đầu vào)
            recon_loss = criterion.recon_loss_fn(recon_outputs, images)

            # Gộp Loss
            loss = seg_loss_total + criterion.recon_weight * recon_loss
        else:
            # Code dự phòng nếu không dùng DS
            loss, dice_score, recon_loss = criterion(outputs, recon_outputs, masks, images, attn_weights)

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

            outputs, recon_outputs, attn_weights = model(images)

            if isinstance(outputs, list):
                outputs = outputs[0]

            # ===== DEBUG PROBABILITY =====
            if i == 0:
                probs = torch.softmax(outputs, dim=1)
                print(f"\n[DEBUG] Max prob: {probs.max().item():.4f}")

            # ===== LOSS =====
            loss, dice_score, _ = criterion(outputs, recon_outputs, masks, images, attn_weights)
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


def save_checkpoint(state, is_best, checkpoint_dir, fold):
    """Cập nhật thêm tham số `fold` để lưu tên file không bị đè nhau"""
    filename = os.path.join(checkpoint_dir, f"last_checkpoint_fold_{fold}.pth")
    torch.save(state, filename)

    if is_best:
        best_filename = os.path.join(checkpoint_dir, f"best_model_fold_{fold}.pth")
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

    # =================================================================
    # 1. TỰ ĐỘNG HÓA K-FOLD SPLITTING TRÊN VOLUME
    # =================================================================
    orig_split_file = cfg['paths']['modal_split_file']
    # Đổi tên file: split.json -> split_kfold.json
    kfold_split_file = orig_split_file.replace(".json", "_kfold.json")
    k_folds = cfg['train'].get('k_fold', 5)

    auto_generate_kfold(orig_split_file, kfold_split_file, k=k_folds)

    # Cập nhật config để Dataset biết đường đọc file mới
    cfg['paths']['modal_split_file'] = kfold_split_file

    # =================================================================
    # 2. VÒNG LẶP HUẤN LUYỆN K-FOLD
    # =================================================================
    for fold in range(k_folds):
        print("\n" + "=" * 50)
        print(f"🔥 ĐANG CHẠY FOLD {fold} / {k_folds - 1}")
        print("=" * 50)

        # Khởi tạo lại Dữ liệu cho từng Fold
        # (Lưu ý: Bạn nhớ cập nhật dataset.py để hỗ trợ truyền biến fold_idx nhé)
        train_ds = TransAttUnetDataset(
            cfg['paths']['modal_processed_data'],
            cfg['paths']['modal_split_file'],
            mode='train',
            fold_idx=fold  # Thêm biến này
        )

        val_ds = TransAttUnetDataset(
            cfg['paths']['modal_processed_data'],
            cfg['paths']['modal_split_file'],
            mode='val',
            fold_idx=fold  # Thêm biến này
        )

        train_loader = DataLoader(
            train_ds, batch_size=cfg['train']['batch_size'],
            shuffle=True, num_workers=cfg['data']['num_workers'], pin_memory=True
        )

        val_loader = DataLoader(
            val_ds, batch_size=cfg['train']['batch_size'],
            shuffle=False, num_workers=cfg['data']['num_workers'], pin_memory=True
        )

        # BẮT BUỘC: Reset lại Model, Optimizer và Scheduler về trạng thái ban đầu cho mỗi Fold
        model = TransAttUnet(
            n_channels=cfg['model']['architecture']['n_channels'],
            n_classes=cfg['model']['architecture']['n_classes']
        ).to(device)

        core_loss = AttentionWeightedFocalTverskyLoss(alpha=0.3, beta=0.7)
        criterion = MultiTaskLoss(seg_loss_fn=core_loss, recon_weight=0.1)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg['train']['optimizer']['lr'],
            weight_decay=cfg['train']['optimizer']['weight_decay']
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg['train']['epochs'], eta_min=1e-6
        )

        # Logic Resume (Hiện tại hỗ trợ resume fold cuối cùng bị crash)
        start_epoch = 1
        best_dice = 0.0

        # Vòng lặp Epoch tiêu chuẩn cho Fold hiện tại
        patience = cfg['train'].get('early_stopping', 20)
        no_improve_count = 0

        for epoch in range(start_epoch, cfg['train']['epochs'] + 1):
            print(f"\n[Fold {fold}] Epoch [{epoch}]")

            train_loss, train_dice, train_class_losses = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )

            val_loss, val_metrics = validate(
                model, val_loader, criterion, device
            )

            scheduler.step()

            print(f"Train Loss: {train_loss:.4f} | Dice: {train_dice:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Dice: {val_metrics['dice']:.4f}")

            # Kiểm tra cải thiện
            is_best = val_metrics['dice'] > best_dice
            if is_best:
                best_dice = val_metrics['dice']
                no_improve_count = 0
                print(f"⭐ New Best Dice for Fold {fold}: {best_dice:.4f}")
            else:
                no_improve_count += 1

            # Lưu checkpoint có đuôi fold
            checkpoint = {
                'epoch': epoch,
                'fold': fold,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dice': best_dice
            }

            if is_best or epoch % cfg['train']['save_interval'] == 0:
                save_checkpoint(checkpoint, is_best, cfg['paths']['checkpoint_dir'], fold)

            if no_improve_count >= patience:
                print(f"🛑 Early stopping Fold {fold} at epoch {epoch}. Best Dice: {best_dice:.4f}")
                break

        print(f"\n✅ Hoàn thành Fold {fold} | Best Dice: {best_dice:.4f}")

    print("\n🎉 HOÀN THÀNH TRAINING TOÀN BỘ K-FOLD!")


if __name__ == "__main__":
    main()