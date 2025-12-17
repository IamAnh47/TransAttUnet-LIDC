import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os


class TransAttUnetDataset(Dataset):
    def __init__(self, data_dir, split_file, mode='train'):
        """
        Dataset chuẩn cho bài toán 2D Segmentation (TransAttUnet).

        Args:
            data_dir: Đường dẫn folder 'processed_segmentation' (chứa subfolder images/ và masks/)
            split_file: Đường dẫn file 'split.json'
            mode: 'train', 'val', hoặc 'test'
        """
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")
        self.mode = mode

        # Load danh sách file từ split.json
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Không tìm thấy file split: {split_file}")

        with open(split_file, 'r') as f:
            splits = json.load(f)

        self.file_list = splits.get(mode, [])

        if not self.file_list:
            print(f"⚠️ Cảnh báo: Tập dữ liệu '{mode}' rỗng!")
        else:
            print(f"✅ Đã load tập '{mode}': {len(self.file_list)} mẫu.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]

        # 1. Tạo đường dẫn
        img_path = os.path.join(self.image_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)

        # 2. Load file .npy
        # Preprocess đã lưu float32 chuẩn hóa [0,1] rồi, nên load lên là dùng được ngay
        image = np.load(img_path).astype(np.float32)  # Shape: (512, 512)
        mask = np.load(mask_path).astype(np.float32)  # Shape: (512, 512)

        # 3. Thêm chiều Channels (Quan trọng cho PyTorch)
        # PyTorch Conv2d yêu cầu input là (Batch, Channel, Height, Width)
        # Ảnh của mình là Grayscale nên Channel = 1
        image = np.expand_dims(image, axis=0)  # (512, 512) -> (1, 512, 512)
        mask = np.expand_dims(mask, axis=0)  # (512, 512) -> (1, 512, 512)

        # 4. Chuyển sang Tensor
        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask)

        return image_tensor, mask_tensor