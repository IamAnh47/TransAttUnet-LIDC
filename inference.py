import os
import torch
import argparse
import yaml
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import csv

# Import các module
from src.model import TransAttUnet
from src.dataset import TransAttUnetDataset
from src.utils import load_config, set_seed, calculate_metrics, AverageMeter


def save_visualization(image, mask, pred, save_path, dice_score):
    img_np = image.squeeze().cpu().numpy()
    mask_np = mask.squeeze().cpu().numpy()
    pred_np = pred.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title("CT Input")
    axes[0].axis('off')

    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis('off')

    masked_pred = np.ma.masked_where(pred_np != 2, pred_np)
    axes[2].imshow(img_np, cmap='gray')
    axes[2].imshow(masked_pred, cmap='autumn', alpha=0.6)
    axes[2].set_title(f"Prediction (Dice: {dice_score:.4f})")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def apply_crf(img_numpy, prob_numpy, num_classes=3):
    C, H, W = prob_numpy.shape
    if img_numpy.max() <= 1.0:
        img_uint8 = (img_numpy * 255).astype(np.uint8)
    else:
        img_uint8 = img_numpy.astype(np.uint8)
    img_rgb = np.stack([img_uint8] * 3, axis=-1)
    img_rgb = np.ascontiguousarray(img_rgb)

    prob_numpy = np.ascontiguousarray(prob_numpy)
    unary = unary_from_softmax(prob_numpy)
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF2D(W, H, C)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseBilateral(sxy=(5, 5), srgb=(13, 13, 13), rgbim=img_rgb, compat=10,
                           kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(5)
    res = np.argmax(Q, axis=0).reshape((H, W))
    return res


def tta_predict(model, images):
    logits_orig, _ = model(images)

    images_hf = torch.flip(images, dims=[3])
    logits_hf, _ = model(images_hf)
    logits_hf = torch.flip(logits_hf, dims=[3])

    images_vf = torch.flip(images, dims=[2])
    logits_vf, _ = model(images_vf)
    logits_vf = torch.flip(logits_vf, dims=[2])

    images_hvf = torch.flip(images, dims=[2, 3])
    logits_hvf, _ = model(images_hvf)
    logits_hvf = torch.flip(logits_hvf, dims=[2, 3])

    avg_logits = (logits_orig + logits_hf + logits_vf + logits_hvf) / 4.0
    return avg_logits


def evaluate_ensemble(models, loader, device, result_dir, vis_count=20):
    # Đưa toàn bộ 5 models về chế độ eval
    for m in models:
        m.eval()

    metrics_meters = {
        "dice": AverageMeter(),
        "iou": AverageMeter(),
        "recall": AverageMeter(),
        "precision": AverageMeter(),
        "accuracy": AverageMeter()
    }

    csv_file = open(os.path.join(result_dir, "test_results.csv"), mode='w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(["Filename", "Nodule_Dice", "IoU", "Recall", "Precision", "Accuracy"])

    vis_dir = os.path.join(result_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    results_list = []

    print(f"Đang chạy Ensemble ({len(models)} models) trên tập Test...")
    with torch.no_grad():
        pbar = tqdm(loader, desc="Testing")
        for i, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)

            # --- SỨC MẠNH ENSEMBLE: CỘNG GỘP LOGITS TỪ 5 MÔ HÌNH ---
            ensemble_logits = 0
            for model in models:
                ensemble_logits += tta_predict(model, images)

            # Chia trung bình logits của 5 mô hình
            avg_logits = ensemble_logits / len(models)
            probs = torch.softmax(avg_logits, dim=1)

            # ÁP DỤNG CRF
            crf_preds = []
            for b in range(images.size(0)):
                img_np = images[b, 0].cpu().numpy()
                prob_np = probs[b].cpu().numpy()
                pred_crf = apply_crf(img_np, prob_np, num_classes=3)
                crf_preds.append(pred_crf)

            crf_tensor = torch.tensor(np.stack(crf_preds), device=device, dtype=torch.long)
            crf_logits = torch.nn.functional.one_hot(crf_tensor, num_classes=3).permute(0, 3, 1, 2).float() * 10.0

            scores = calculate_metrics(crf_logits, masks)

            # LẤY RIÊNG CHỈ SỐ CỦA KHỐI U (CLASS 2)
            dice_nodule = scores['dice_per_class'][2]
            scores['dice'] = dice_nodule

            for k, v in scores.items():
                if k == 'dice_per_class' or k == 'iou_per_class': continue
                if k not in metrics_meters: metrics_meters[k] = AverageMeter()
                metrics_meters[k].update(v, images.size(0))

            writer.writerow([f"Test_Sample_{i}", dice_nodule, scores['iou'],
                             scores['recall'], scores['precision'], scores['accuracy']])

            pred_mask = crf_tensor.float()
            if images.size(0) == 1:
                has_nodule_gt = (masks == 2).sum() > 0
                has_nodule_pred = (pred_mask == 2).sum() > 0

                if has_nodule_gt or has_nodule_pred:
                    results_list.append({
                        'index': i,
                        'dice': float(dice_nodule),
                        'image': images[0].cpu().clone(),
                        'mask': masks[0].cpu().clone(),
                        'pred': pred_mask[0].cpu().clone()
                    })

    csv_file.close()

    # --- XUẤT ẢNH TOP 20 TỐT NHẤT / TỆ NHẤT ---
    print("\nĐang xuất ảnh visualizations (Tốt nhất & Tệ nhất)...")
    results_list.sort(key=lambda x: x['dice'])
    num_to_save = min(vis_count, len(results_list))

    # Tệ nhất
    for rank, item in enumerate(results_list[:num_to_save]):
        save_path = os.path.join(vis_dir, f"WORST_{rank + 1}_idx_{item['index']}_Dice_{item['dice']:.4f}.png")
        save_visualization(item['image'], item['mask'], item['pred'], save_path, item['dice'])

    # Tốt nhất
    for rank, item in enumerate(reversed(results_list[-num_to_save:])):
        save_path = os.path.join(vis_dir, f"BEST_{rank + 1}_idx_{item['index']}_Dice_{item['dice']:.4f}.png")
        save_visualization(item['image'], item['mask'], item['pred'], save_path, item['dice'])

    return {k: v.avg for k, v in metrics_meters.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    # Đã bỏ --model_path vì giờ nạp tự động cả 5 fold
    parser.add_argument("--vis_num", type=int, default=20, help="Số lượng ảnh visualize mỗi loại")
    parser.add_argument("--save_dir", type=str, default="/mnt/storage/results_roi")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg['train']['device'] if torch.cuda.is_available() else "cpu")

    RESULT_DIR = args.save_dir
    os.makedirs(RESULT_DIR, exist_ok=True)

    # 1. Load Data (Tập Test)
    test_ds = TransAttUnetDataset(
        cfg['paths']['modal_processed_data'],
        cfg['paths']['modal_split_file'],
        mode='test'
    )

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=cfg['data']['num_workers'], pin_memory=True)
    print(f"Đã load tập Test: {len(test_ds)} mẫu.")

    # 2. Load CẢ 5 MÔ HÌNH (Ensemble)
    models = []
    k_folds = cfg['train'].get('k_fold', 5)
    checkpoint_dir = cfg['paths']['checkpoint_dir']

    for fold in range(k_folds):
        model_path = os.path.join(checkpoint_dir, f"best_model_fold_{fold}.pth")
        if not os.path.exists(model_path):
            print(f"⚠️ Cảnh báo: Không tìm thấy {model_path}. Hãy chắc chắn bạn đã train xong fold này!")
            continue

        print(f"Đang load trọng số: Fold {fold}")
        model = TransAttUnet(
            n_channels=cfg['model']['architecture']['n_channels'],
            n_classes=cfg['model']['architecture']['n_classes']
        ).to(device)

        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        models.append(model)

    if len(models) == 0:
        print("❌ Lỗi: Không có mô hình nào được load. Hủy đánh giá.")
        return

    # 3. Chạy Đánh giá Ensemble
    avg_metrics = evaluate_ensemble(models, test_loader, device, RESULT_DIR, vis_count=args.vis_num)

    # 4. In kết quả
    summary_path = os.path.join(RESULT_DIR, "summary_metrics.txt")
    print("\n" + "=" * 30)
    print("KẾT QUẢ ĐÁNH GIÁ K-FOLD ENSEMBLE (TEST SET)")
    print("=" * 30)
    with open(summary_path, "w") as f:
        f.write("Evaluation Summary (Ensemble)\n")
        f.write("==================\n")
        for k, v in avg_metrics.items():
            line = f"{k.capitalize()}: {v:.4f}"
            print(line)
            f.write(line + "\n")


if __name__ == "__main__":
    main()