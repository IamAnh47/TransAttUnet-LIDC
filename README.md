# TransAttUnet: Lung Nodule Segmentation on LIDC-IDRI

Triển khai mô hình TransAttUnet (Multi-level Attention-guided U-Net with Transformer) để phân vùng nốt phổi trên tập dữ liệu CT phổi LIDC-IDRI.

Dự án được xây dựng bằng PyTorch, bao gồm quy trình xử lý dữ liệu 3D sang 2D, module Self-aware Attention (SAA) và chiến lược huấn luyện tối ưu cho dữ liệu y tế.

1. Kiến trúc và cách hoạt động của mô hình

Mô hình TransAttUnet khắc phục hạn chế của U-Net truyền thống chỉ nhìn cục bộ bằng cách kết hợp cơ chế Attention toàn cục. Kiến trúc gồm 3 phần chính:

A. Encoder (Feature Extraction)

- Sử dụng kiến trúc CNN nhiều tầng để trích xuất đặc trưng từ ảnh CT $512 \times 512$.

- Giảm kích thước không gian và tăng chiều sâu (channels) để nắm bắt ngữ nghĩa cấp cao.

B. Self-aware Attention (SAA) Bridge

Cầu nối giữa Encoder và Decoder. Nó gồm 2 nhánh song song:

Transformer Self Attention (TSA): Giúp mô hình nhìn toàn bộ bức ảnh cùng lúc để nắm bắt mối quan hệ tầm xa, ví dụ liên kết giữa các vùng phổi khác nhau.

Global Spatial Attention (GSA): Mã hóa thông tin vị trí không gian, giúp mô hình định vị chính xác vùng nốt phổi.

C. Decoder & Multi-scale Skip Connections

- Khôi phục kích thước ảnh về $512 \times 512$ để tạo mặt nạ (mask).

- Sử dụng Residual Skip Connections: Thay vì nối ghép đơn giản, mô hình cộng gộp thông tin từ các tầng Encoder để giữ lại chi tiết biên cạnh sắc nét của khối u.

2. Cài đặt môi trường

Đảm bảo đã cài đặt Python 3.8+.
```bash
    git clone https://github.com/yourusername/TransAttUnet-LIDC.git
    cd TransAttUnet-LIDC
    python -m venv venv# Windows:
    venv\Scripts\activate
    
    pip install -r requirements.txt
```
3. Chuẩn bị dữ liệu & Preprocessing

Dữ liệu đầu vào là tập LIDC-IDRI (định dạng DICOM). Quy trình Preprocessing được thiết kế để chuyển đổi dữ liệu 3D phức tạp thành ảnh 2D sạch cho mô hình.

Các bước Preprocessing chi tiết:

- Resampling: Đồng nhất độ phân giải không gian về $1mm \times 1mm \times 1mm$ (Isotropic) để tránh méo hình.

- Intensity Windowing: Áp dụng Lung Window để làm rõ nhu mô phổi:

Window Center: -600 HU

Window Width: 1500 HU

- Chuẩn hóa giá trị pixel về khoảng [0, 1].

- Consensus Masking: Tạo nhãn (Ground Truth) từ 4 bác sĩ. Sử dụng mức đồng thuận 50% (ít nhất 2 bác sĩ đồng ý là nốt phổi).

- Slicing và Filtering: Cắt khối 3D thành các lát cắt 2D ($512 \times 512$). Chỉ giữ lại các lát cắt có chứa nốt phổi (Positive Slices).

Cách chạy Preprocessing:
```bash
    # Xử lý toàn bộ dữ liệu
    
    python preprocess.py
    
    python preprocess.py --num 10
    
    tar -czf processed.tar.gz data/processed
```

Nếu train trên modal thì chạy cell giải nén để train cho dễ

```bash
    import os
    import shutil
    
    volume_root = "/mnt/TransAttUnet"
    
    # 1. Đảm bảo có folder data/processed
    target_dir = os.path.join(volume_root, "data", "processed")
    os.makedirs(target_dir, exist_ok=True)
    print(f"Đã sẵn sàng thư mục: {target_dir}")
    
    # 2. Giải nén processed.tar.gz vào đúng data/processed
    tar_path = os.path.join(volume_root, "processed.tar.gz")
    
    if not os.path.exists(tar_path):
        print("Không tìm thấy file processed.tar.gz – bạn đã upload chưa?")
    else:
        print("Đang giải nén processed.tar.gz → data/processed ... (chờ 20–60 giây)")
        !tar -xzf "{tar_path}" -C "{target_dir}" --strip-components=0
        print("GIẢI NÉN XONG 100%!")
    
    # 3. Kiểm tra kết quả
    print("\nN{party popper} HOÀN TẤT! Cấu trúc hiện tại:")
    print("data/processed chứa:")
    print(os.listdir(target_dir)[:30]) 
    
    #if os.path.exists(tar_path):
    #    os.remove(tar_path)
    #    print(f"\nĐã xóa {tar_path} để tiết kiệm dung lượng")
    
    print("\nSẴN SÀNG TRAIN MODEL!")
```

Kết quả sẽ được lưu vào thư mục data/processed/ gồm:

images/: File .npy chứa ảnh CT.

masks/: File .npy chứa nhãn phân vùng.

split.json: File chia tập Train/Val/Test.

4. Kiểm tra dữ liệu

Trước khi train, hãy kiểm tra xem ảnh và mask có khớp nhau không:

```bash
    python check_data.py
```
-> Hiển thị ngẫu nhiên bộ 3 hình: Ảnh gốc - Mask - Ảnh chồng (Overlay) để đánh giá chất lượng.

5. Training

Mô hình được huấn luyện với hàm loss kết hợp: Loss = 0.5 * BCE + 0.5 * Dice.

    Cấu hình (configs/config.yaml)
    
    Có thể thay đổi các tham số quan trọng:
    
    batch_size: Mặc định 4 (Tăng lên 8/16 nếu VRAM > 12GB).
    
    lr (Learning Rate): Mặc định 0.001 (cho SGD).
    
    epochs: Mặc định 100.
```bash
    python train.py
```
Tiếp tục huấn luyện (Resume):

Nếu quá trình train bị ngắt quãng, có thể chạy tiếp từ checkpoint gần nhất:
```bash
    python train.py --resume outputs/checkpoints/last_checkpoint.pth
```