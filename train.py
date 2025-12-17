import os
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim

# Import các module chúng ta đã viết
from src.model import TransAttUnet
from src.dataset import TransAttUnetDataset
from src.loss import TransAttLoss
from src.utils import load_config, set_seed, calculate_metrics, AverageMeter


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()

    pbar = tqdm(loader, desc="Training", leave=False)

    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss, dice_score = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))
        dice_meter.update(dice_score.item(), images.size(0))

        pbar.set_postfix({"Loss": f"{loss_meter.avg:.4f}", "Dice": f"{dice_meter.avg:.4f}"})

    return loss_meter.avg, dice_meter.avg


def validate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    metrics_meters = {
        "dice": AverageMeter(), "iou": AverageMeter(),
        "accuracy": AverageMeter(), "recall": AverageMeter(), "precision": AverageMeter()
    }

    with torch.no_grad():
        for i, (images, masks) in tqdm(enumerate(loader), desc="Validating",
                                       leave=False):  # Sửa lại enumerate để lấy index
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            if i == 0:
                probs = torch.sigmoid(outputs)
                print(f"\n[DEBUG] Max predict prob: {probs.max().item():.4f} (Threshold is 0.5)")
                # Nếu Max prob < 0.5 -> Mô hình chưa dám tô màu nốt phổi nào
            loss, _ = criterion(outputs, masks)
            loss_meter.update(loss.item(), images.size(0))

            scores = calculate_metrics(outputs, masks)
            for k, v in scores.items():
                metrics_meters[k].update(v, images.size(0))

    return loss_meter.avg, {k: v.avg for k, v in metrics_meters.items()}


def save_checkpoint(state, is_best, checkpoint_dir):
    """Hàm lưu checkpoint an toàn"""
    # Lưu file định kỳ (resume được)
    filename = os.path.join(checkpoint_dir, "last_checkpoint.pth")
    torch.save(state, filename)

    # Nếu là best model, lưu thêm 1 bản riêng (chỉ cần weights để inference)
    if is_best:
        best_filename = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(state['model_state_dict'], best_filename)


def main():
    # 1. Setup Config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")  # <--- MỚI
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg['train']['seed'])

    os.makedirs(cfg['paths']['checkpoint_dir'], exist_ok=True)
    device = torch.device(cfg['train']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Start training on: {device}")

    # 2. Prepare Data
    train_ds = TransAttUnetDataset(cfg['paths']['processed_data'], cfg['paths']['split_file'], mode='train')
    val_ds = TransAttUnetDataset(cfg['paths']['processed_data'], cfg['paths']['split_file'], mode='val')

    train_loader = DataLoader(train_ds, batch_size=cfg['train']['batch_size'],
                              shuffle=True, num_workers=cfg['data']['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['train']['batch_size'],
                            shuffle=False, num_workers=cfg['data']['num_workers'], pin_memory=True)

    # 3. Model, Loss, Optimizer
    model = TransAttUnet(
        n_channels=cfg['model']['architecture']['n_channels'],
        n_classes=cfg['model']['architecture']['n_classes']
    ).to(device)

    criterion = TransAttLoss(alpha=cfg['train']['loss']['alpha'], beta=cfg['train']['loss']['beta'])

    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg['train']['optimizer']['lr'],
        momentum=cfg['train']['optimizer']['momentum'],
        weight_decay=cfg['train']['optimizer']['weight_decay']
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg['train']['scheduler']['step_size'],
        gamma=cfg['train']['scheduler']['gamma']
    )

    # --- KHÔI PHỤC CHECKPOINT (RESUME LOGIC) ---
    start_epoch = 1
    best_dice = 0.0

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)

            # Load states
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # Khôi phục epoch và best score
            start_epoch = checkpoint['epoch'] + 1
            best_dice = checkpoint['best_dice']
            print(f"Resumed from epoch {checkpoint['epoch']} with Best Dice: {best_dice:.4f}")
        else:
            print(f"No checkpoint found at: {args.resume}")

    # 4. Training Loop
    total_epochs = cfg['train']['epochs']

    for epoch in range(start_epoch, total_epochs + 1):
        print(f"\nEpoch [{epoch}/{total_epochs}] | LR: {optimizer.param_groups[0]['lr']:.6f}")

        train_loss, train_dice = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(f"   Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        print(f"   Val Loss:   {val_loss:.4f} | Val Dice:   {val_metrics['dice']:.4f} | IoU: {val_metrics['iou']:.4f}")

        # --- LƯU CHECKPOINT ĐẦY ĐỦ ---
        is_best = val_metrics['dice'] > best_dice
        if is_best:
            best_dice = val_metrics['dice']
            print(f"   New Best Model! (Dice: {best_dice:.4f})")

        # Tạo dict chứa toàn bộ trạng thái
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_dice': best_dice
        }

        # Lưu file
        if is_best or epoch % cfg['train']['save_interval'] == 0:
            save_checkpoint(checkpoint_dict, is_best, cfg['paths']['checkpoint_dir'])

    print("\nTraining Completed!")
    print(f"Best Validation Dice Score: {best_dice:.4f}")


if __name__ == "__main__":
    main()