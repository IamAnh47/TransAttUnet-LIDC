import os
import torch
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import csv

# Import các module
from src.model import TransAttUnet
from src.dataset import TransAttUnetDataset
from src.utils import load_config, set_seed, calculate_metrics, AverageMeter


def save_visualization(image, mask, pred, save_path, dice_score):
    """
    Lưu ảnh so sánh: Input | Ground Truth | Prediction
    """
    # Chuyển tensor về numpy và remove batch dimension
    # image: (1, 512, 512) -> (512, 512)
    img_np = image.squeeze().cpu().numpy()
    mask_np = mask.squeeze().cpu().numpy()
    pred_np = pred.squeeze().cpu().numpy()  # Đã qua sigmoid và threshold

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Ảnh gốc CT
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title("CT Input")
    axes[0].axis('off')

    # 2. Ground Truth
    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis('off')

    # 3. Prediction (Overlay lên ảnh gốc cho dễ nhìn)
    # Mask dự đoán màu đỏ bán trong suốt
    masked_pred = np.ma.masked_where(pred_np == 0, pred_np)
    axes[2].imshow(img_np, cmap='gray')
    axes[2].imshow(masked_pred, cmap='autumn', alpha=0.6)
    axes[2].set_title(f"Prediction (Dice: {dice_score:.4f})")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def evaluate(model, loader, device, result_dir, vis_count=20):
    model.eval()

    # Metrics meters
    metrics_meters = {
        "dice": AverageMeter(),
        "iou": AverageMeter(),
        "recall": AverageMeter(),
        "precision": AverageMeter(),
        "accuracy": AverageMeter()
    }

    # CSV file để lưu kết quả từng ảnh
    csv_file = open(os.path.join(result_dir, "test_results.csv"), mode='w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(["Filename", "Dice", "IoU", "Recall", "Precision", "Accuracy"])

    vis_dir = os.path.join(result_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    print("Đang chạy đánh giá trên tập Test...")

    count_vis = 0

    with torch.no_grad():
        # Dùng enumerate để lấy index, nhưng dataset của chúng ta trả về (image, mask)
        # Để lấy tên file, ta cần sửa Dataset một chút hoặc chấp nhận không có tên file trong CSV
        # Tuy nhiên, TransAttUnetDataset hiện tại chưa trả về filename.
        # Tạm thời ta dùng index.

        pbar = tqdm(loader, desc="Testing")
        for i, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)

            # Forward
            logits = model(images)

            # Tính metrics
            scores = calculate_metrics(logits, masks)

            # Update average meters
            for k, v in scores.items():
                metrics_meters[k].update(v, images.size(0))

            # Ghi vào CSV
            # Giả sử batch_size = 1 cho inference để dễ xử lý từng ảnh
            writer.writerow([f"Test_Sample_{i}", scores['dice'], scores['iou'],
                             scores['recall'], scores['precision'], scores['accuracy']])

            # Lưu ảnh visualize (Chỉ lưu vis_count ảnh đầu tiên hoặc ảnh có Dice > 0 để đỡ tốn chỗ)
            # Logic: Lưu 10 ảnh đầu tiên + 10 ảnh có Dice > 0.5 tiếp theo
            pred_mask = (torch.sigmoid(logits) > 0.5).float()

            # Chỉ visualize nếu batch_size = 1 (để code đơn giản)
            if images.size(0) == 1 and count_vis < vis_count:
                # Chỉ lưu những ảnh có nốt phổi (GT có pixel > 0) hoặc Model đoán ra cái gì đó
                if masks.sum() > 0 or pred_mask.sum() > 0:
                    save_path = os.path.join(vis_dir, f"result_{i}_dice_{scores['dice']:.3f}.png")
                    save_visualization(images[0], masks[0], pred_mask[0], save_path, scores['dice'])
                    count_vis += 1

    csv_file.close()

    return {k: v.avg for k, v in metrics_meters.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Đường dẫn file .pth (mặc định lấy best_model.pth trong output)")
    parser.add_argument("--vis_num", type=int, default=50, help="Số lượng ảnh muốn lưu visualize")
    args = parser.parse_args()

    # 1. Load Config
    cfg = load_config(args.config)
    device = torch.device(cfg['train']['device'] if torch.cuda.is_available() else "cpu")

    # Xác định đường dẫn model
    if args.model_path is None:
        model_path = os.path.join(cfg['paths']['checkpoint_dir'], "best_model.pth")
    else:
        model_path = args.model_path

    if not os.path.exists(model_path):
        print(f"Không tìm thấy model tại: {model_path}")
        return

    # 2. Setup Output Dir
    RESULT_DIR = "results"
    os.makedirs(RESULT_DIR, exist_ok=True)

    # 3. Load Data (Tập Test)
    test_ds = TransAttUnetDataset(
        cfg['paths']['processed_data'],
        cfg['paths']['split_file'],
        mode='test'
    )

    # Batch size = 1 để dễ visualize từng ảnh và tính chỉ số chi tiết
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=cfg['data']['num_workers'], pin_memory=True)

    print(f"Đã load tập Test: {len(test_ds)} mẫu.")

    # 4. Load Model
    print(f"Đang load model từ: {model_path}")
    model = TransAttUnet(
        n_channels=cfg['model']['architecture']['n_channels'],
        n_classes=cfg['model']['architecture']['n_classes']
    ).to(device)

    # Load weights (xử lý trường hợp lưu cả dict checkpoint hoặc chỉ state_dict)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)  # Trường hợp chỉ lưu weights

    # 5. Run Evaluation
    avg_metrics = evaluate(model, test_loader, device, RESULT_DIR, vis_count=args.vis_num)

    # 6. In kết quả và lưu file tổng hợp
    summary_path = os.path.join(RESULT_DIR, "summary_metrics.txt")

    print("\n" + "=" * 30)
    print("KẾT QUẢ ĐÁNH GIÁ (TEST SET)")
    print("=" * 30)
    with open(summary_path, "w") as f:
        f.write("Evaluation Summary\n")
        f.write("==================\n")
        for k, v in avg_metrics.items():
            line = f"{k.capitalize()}: {v:.4f}"
            print(line)
            f.write(line + "\n")

    print("=" * 30)
    print(f"Kết quả chi tiết đã lưu tại: {RESULT_DIR}/test_results.csv")
    print(f"Ảnh visualize đã lưu tại: {RESULT_DIR}/visualizations/")


if __name__ == "__main__":
    main()