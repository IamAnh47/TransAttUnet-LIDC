import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def test_clahe_on_ct(npy_path):
    if not os.path.exists(npy_path):
        print(f"Không tìm thấy file: {npy_path}")
        return

    # 1. Load ảnh CT gốc (đang ở dạng float32)
    image = np.load(npy_path).astype(np.float32)

    # 2. Chuyển đổi sang dải 0-255 (uint8) vì thuật toán cv2 yêu cầu
    # (cv2.normalize sẽ tự động tìm min-max trong ảnh và kéo giãn nó ra)
    img_uint8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 3. Tạo bộ lọc CLAHE
    # clipLimit: Giới hạn độ gắt của tương phản (càng cao càng gắt, 2.0 - 4.0 là đẹp cho y tế)
    # tileGridSize: Kích thước ô lưới cục bộ
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # 4. Áp dụng CLAHE lên ảnh
    image_clahe = clahe.apply(img_uint8)

    # 5. Visualize so sánh (Ảnh gốc vs Ảnh CLAHE)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(img_uint8, cmap='gray')
    axes[0].set_title("CT Gốc (Tua gai mờ nhạt)")
    axes[0].axis('off')

    axes[1].imshow(image_clahe, cmap='gray')
    axes[1].set_title("Sau khi bật kính lúp CLAHE (Chi tiết nổi bật)")
    axes[1].axis('off')

    plt.tight_layout()

    # Lưu ra file hoặc show trực tiếp
    plt.savefig("clahe_comparison.png")
    print("Đã lưu ảnh so sánh tại: clahe_comparison.png")
    # plt.show() # Mở comment dòng này nếu bạn chạy trên máy có màn hình (local)


if __name__ == "__main__":
    # Thay đường dẫn này bằng 1 file .npy bất kỳ trong tập train/val của bạn
    # Ví dụ: /mnt/storage/data/images/0001.npy
    SAMPLE_IMAGE_PATH = ".\data\processed\images\LIDC-IDRI-0017_nodule0_slice160_714c25.npy"

    test_clahe_on_ct(SAMPLE_IMAGE_PATH)