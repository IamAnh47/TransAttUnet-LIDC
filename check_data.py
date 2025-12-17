import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random
import sys

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Đường dẫn đến folder dữ liệu đã xử lý
PROCESSED_DIR = "data/processed"
IMG_DIR = os.path.join(PROCESSED_DIR, "images")
MASK_DIR = os.path.join(PROCESSED_DIR, "masks")


def visualize_slice(filename):
    """
    Hiển thị ảnh CT, Mask và Overlay từ file .npy
    """
    # Xử lý nếu người dùng nhập đường dẫn đầy đủ thay vì tên file
    filename = os.path.basename(filename)

    img_path = os.path.join(IMG_DIR, filename)
    mask_path = os.path.join(MASK_DIR, filename)

    # 1. Kiểm tra file tồn tại
    if not os.path.exists(img_path):
        print(f"Không tìm thấy ảnh: {filename}")
        print(f"   (Đang tìm trong: {IMG_DIR})")
        return

    if not os.path.exists(mask_path):
        print(f"CẢNH BÁO: Có ảnh nhưng mất mask tương ứng! ({filename})")
        return

    # 2. Load dữ liệu
    img = np.load(img_path)
    mask = np.load(mask_path)

    # 3. In thông số thống kê
    print(f"\nĐang kiểm tra: {filename}")
    print(f"   - Shape Image: {img.shape} | Shape Mask: {mask.shape}")
    print(f"   - Image Range: [{img.min():.4f}, {img.max():.4f}]")
    print(f"   - Mask Values: {np.unique(mask)}")

    # 4. Kiểm tra hợp lệ (Sanity Check)
    warnings = []
    if img.shape != (512, 512):
        warnings.append(f"Kích thước sai (Chuẩn bài báo là 512x512, thực tế {img.shape})")

    if img.min() < 0 or img.max() > 1.05:  # Cho phép sai số nhỏ do float
        warnings.append("Ảnh chưa được chuẩn hóa về [0, 1] (Windowing có vấn đề?)")

    if not np.array_equal(np.unique(mask), [0, 1]) and not np.array_equal(np.unique(mask), [0]):
        # Mask có thể toàn 0 (nền đen), nhưng nếu có giá trị khác 0 và 1 là sai
        warnings.append(f"Mask chứa giá trị lạ (không phải Binary 0/1): {np.unique(mask)}")

    if warnings:
        for w in warnings: print(w)
    else:
        print("Dữ liệu HỢP LỆ (Ready for TransAttUnet).")

    # 5. Vẽ hình (Visualization)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Hình 1: Ảnh CT gốc
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("CT Image (Lung Window)")
    axes[0].axis('off')

    # Hình 2: Mask (Ground Truth)
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Segmentation Mask")
    axes[1].axis('off')

    # Hình 3: Overlay (Chồng mask lên ảnh)
    # Tạo mask bán trong suốt màu đỏ
    masked_img = np.ma.masked_where(mask == 0, mask)

    axes[2].imshow(img, cmap='gray')
    axes[2].imshow(masked_img, cmap='autumn', alpha=0.5)  # Mask màu đỏ/cam
    axes[2].set_title("Overlay (Kiểm tra biên)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Cách dùng 1: python check_data.py <tên_file.npy>
    if len(sys.argv) > 1:
        target = sys.argv[1]
        visualize_slice(target)

    # Cách dùng 2: python check_data.py (Tự chọn ngẫu nhiên)
    else:
        files = glob.glob(os.path.join(IMG_DIR, "*.npy"))

        if not files:
            print(f"Không tìm thấy dữ liệu nào trong {IMG_DIR}")
            print("Đã chạy preprocess.py chưa?")
        else:
            print(f"Tìm thấy {len(files)} lát cắt (slices).")
            while True:
                target_file = random.choice(files)
                visualize_slice(target_file)

                ans = input("Bạn có muốn xem lát cắt khác không? (y/n): ")
                if ans.lower() != 'y':
                    break