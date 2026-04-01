import os
import torch
import argparse
import yaml
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
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
    masked_pred = np.ma.masked_where(pred_np != 2, pred_np)
    axes[2].imshow(img_np, cmap='gray')
    axes[2].imshow(masked_pred, cmap='autumn', alpha=0.6)
    axes[2].set_title(f"Prediction (Dice: {dice_score:.4f})")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def apply_crf(img_numpy, prob_numpy, num_classes=3):
    """
    img_numpy: Ảnh gốc kích thước (H, W), giá trị đã chuẩn hóa 0-1
    prob_numpy: Xác suất dự đoán từ model kích thước (C, H, W)
    """
    C, H, W = prob_numpy.shape

    # 1. Chuyển ảnh CT (grayscale) sang dạng RGB (3 kênh giống nhau) và uint8 vì pydensecrf yêu cầu thế
    if img_numpy.max() <= 1.0:
        img_uint8 = (img_numpy * 255).astype(np.uint8)
    else:
        img_uint8 = img_numpy.astype(np.uint8)
    img_rgb = np.stack([img_uint8] * 3, axis=-1)
    img_rgb = np.ascontiguousarray(img_rgb)

    # 2. Xử lý xác suất (Unary potential)
    prob_numpy = np.ascontiguousarray(prob_numpy)
    unary = unary_from_softmax(prob_numpy)
    unary = np.ascontiguousarray(unary)

    # 3. Khởi tạo CRF
    d = dcrf.DenseCRF2D(W, H, C)
    d.setUnaryEnergy(unary)

    # 4. Thêm điều kiện Không gian (Smoothness) - Xóa nhiễu vụn vặt
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    # 5. Thêm điều kiện Cạnh/Viền (Bilateral) - Đây là ma thuật kéo viền bám vào gai khối u!
    # sxy: Khoảng cách không gian, srgb: Độ nhạy với sự thay đổi màu/độ sáng của pixel
    d.addPairwiseBilateral(sxy=(5, 5), srgb=(13, 13, 13), rgbim=img_rgb, compat=10,
                           kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    # 6. Chạy suy luận CRF (5 bước lặp là đủ hội tụ)
    Q = d.inference(5)

    # 7. Lấy class có xác suất cao nhất sau khi tinh chỉnh
    res = np.argmax(Q, axis=0).reshape((H, W))

    return res

def tta_predict(model, images):
    """
    Test-Time Augmentation (TTA)
    (Đã cập nhật để tương thích với model có Attention Weights)
    """
    # Unpack tuple: Lấy logits, bỏ qua attn_weights bằng dấu '_'
    logits_orig, _ = model(images)

    # Horizontal
    images_hf = torch.flip(images, dims=[3])
    logits_hf, _ = model(images_hf)
    logits_hf = torch.flip(logits_hf, dims=[3])  # Lật mask dự đoán ngược lại

    # Vertical
    images_vf = torch.flip(images, dims=[2])
    logits_vf, _ = model(images_vf)
    logits_vf = torch.flip(logits_vf, dims=[2])  # Lật mask dự đoán ngược lại

    # hyper
    images_hvf = torch.flip(images, dims=[2, 3])
    logits_hvf, _ = model(images_hvf)
    logits_hvf = torch.flip(logits_hvf, dims=[2, 3])  # Lật mask dự đoán ngược lại

    # avg
    avg_logits = (logits_orig + logits_hf + logits_vf + logits_hvf) / 4.0

    return avg_logits


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
    writer.writerow(["Filename", "Nodule_Dice", "IoU", "Recall", "Precision", "Accuracy"])

    vis_dir = os.path.join(result_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    print("Đang chạy đánh giá trên tập Test...")

    # --- KHỞI TẠO LIST LƯU KẾT QUẢ ĐỂ SORT ---
    results_list = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Testing")
        for i, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)

            # Forward với TTA
            logits = tta_predict(model, images)
            probs = torch.softmax(logits, dim=1)

            # 2. ÁP DỤNG CRF LÀM SẮC NÉT ĐƯỜNG VIỀN
            crf_preds = []
            for b in range(images.size(0)):
                img_np = images[b, 0].cpu().numpy()
                prob_np = probs[b].cpu().numpy()

                pred_crf = apply_crf(img_np, prob_np, num_classes=3)
                crf_preds.append(pred_crf)

            crf_tensor = torch.tensor(np.stack(crf_preds), device=device, dtype=torch.long)
            crf_logits = torch.nn.functional.one_hot(crf_tensor, num_classes=3).permute(0, 3, 1, 2).float() * 10.0

            # 4. Tính metrics
            scores = calculate_metrics(crf_logits, masks)

            # LẤY RIÊNG CHỈ SỐ CỦA KHỐI U (CLASS 2)
            dice_nodule = scores['dice_per_class'][2]
            scores['dice'] = dice_nodule

            # Update average meters
            for k, v in scores.items():
                if k == 'dice_per_class' or k == 'iou_per_class':
                    continue
                if k not in metrics_meters:
                    metrics_meters[k] = AverageMeter()
                metrics_meters[k].update(v, images.size(0))

            # Ghi vào CSV
            writer.writerow([f"Test_Sample_{i}", dice_nodule, scores['iou'],
                             scores['recall'], scores['precision'], scores['accuracy']])

            # --- CHUẨN BỊ DỮ LIỆU ĐỂ VISUALIZE ---
            pred_mask = crf_tensor.float()

            if images.size(0) == 1:
                has_nodule_gt = (masks == 2).sum() > 0
                has_nodule_pred = (pred_mask == 2).sum() > 0

                # Chỉ đưa vào danh sách nếu ảnh đó thực sự có u (ở nhãn gốc hoặc dự đoán)
                if has_nodule_gt or has_nodule_pred:
                    results_list.append({
                        'index': i,
                        'dice': float(dice_nodule),
                        # clone và đưa về cpu để tránh tràn RAM GPU
                        'image': images[0].cpu().clone(),
                        'mask': masks[0].cpu().clone(),
                        'pred': pred_mask[0].cpu().clone()
                    })

    csv_file.close()

    # ==========================================================
    # LỌC VÀ LƯU TOP 20 TỐT NHẤT & TOP 20 TỆ NHẤT
    # ==========================================================
    print("\nĐang xuất ảnh visualizations (20 Tốt nhất & 20 Tệ nhất)...")

    # Sắp xếp danh sách từ Dice thấp nhất -> Dice cao nhất
    results_list.sort(key=lambda x: x['dice'])

    num_to_save = min(vis_count, len(results_list))

    # 1. Lưu Top Tệ nhất (Đầu danh sách)
    for rank, item in enumerate(results_list[:num_to_save]):
        save_path = os.path.join(vis_dir, f"WORST_{rank + 1}_idx_{item['index']}_Dice_{item['dice']:.4f}.png")
        save_visualization(item['image'], item['mask'], item['pred'], save_path, item['dice'])

    # 2. Lưu Top Tốt nhất (Cuối danh sách, duyệt ngược lại để lấy từ 1.0 xuống)
    for rank, item in enumerate(reversed(results_list[-num_to_save:])):
        save_path = os.path.join(vis_dir, f"BEST_{rank + 1}_idx_{item['index']}_Dice_{item['dice']:.4f}.png")
        save_visualization(item['image'], item['mask'], item['pred'], save_path, item['dice'])

    return {k: v.avg for k, v in metrics_meters.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Đường dẫn file .pth (mặc định lấy best_model.pth trong output)")
    parser.add_argument("--vis_num", type=int, default=50, help="Số lượng ảnh muốn lưu visualize")
    parser.add_argument("--save_dir", type=str, default="/mnt/storage/results_roi", help="Thư mục lưu kết quả trên Modal Volume")
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
    RESULT_DIR = args.save_dir
    os.makedirs(RESULT_DIR, exist_ok=True)

    # 3. Load Data (Tập Test)
    test_ds = TransAttUnetDataset(
        cfg['paths']['modal_processed_data'],
        cfg['paths']['modal_split_file'],
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