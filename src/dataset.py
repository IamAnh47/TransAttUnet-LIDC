# import torch
# from torch.utils.data import Dataset
# import numpy as np
# import json
# import os
# import random
#
#
# class TransAttUnetDataset(Dataset):
#     def __init__(self, data_dir, split_file, mode='train'):
#         """
#         Data Augmentation Overfitting.
#         """
#         self.data_dir = data_dir
#         self.image_dir = os.path.join(data_dir, "images")
#         self.mask_dir = os.path.join(data_dir, "masks")
#         self.mode = mode
#
#         if not os.path.exists(split_file):
#             raise FileNotFoundError(f"Không tìm thấy file split: {split_file}")
#
#         with open(split_file, 'r') as f:
#             splits = json.load(f)
#
#         self.file_list = splits.get(mode, [])
#
#         if not self.file_list:
#             print(f"Cảnh báo: Tập dữ liệu '{mode}' rỗng!")
#         else:
#             print(f"Đã load tập '{mode}': {len(self.file_list)} mẫu.")
#
#     def __len__(self):
#         return len(self.file_list)
#
#     def augment(self, image, mask):
#         """
#         Hàm thực hiện Augmentation thủ công bằng Numpy.
#         """
#         # Random Flip Horizontal (Lật ngang)
#         if random.random() > 0.5:
#             image = np.flip(image, axis=1)  # Axis 1 là chiều rộng (512)
#             mask = np.flip(mask, axis=1)
#
#         # Random Flip Vertical (Lật dọc)
#         if random.random() > 0.5:
#             image = np.flip(image, axis=0)  # Axis 0 là chiều cao (512)
#             mask = np.flip(mask, axis=0)
#
#         # Random Rotation (Xoay 90, 180, 270 độ)
#         k = random.randint(0, 3)  # 0: 0 độ, 1: 90 độ, 2: 180 độ, 3: 270 độ
#         if k > 0:
#             image = np.rot90(image, k)
#             mask = np.rot90(mask, k)
#
#         return image, mask
#
#     def __getitem__(self, idx):
#         fname = self.file_list[idx]
#
#         img_path = os.path.join(self.image_dir, fname)
#         mask_path = os.path.join(self.mask_dir, fname)
#
#         # Load file .npy
#         # Image shape: (512, 512)
#         image = np.load(img_path).astype(np.float32)
#         mask = np.load(mask_path).astype(np.float32)
#
#         # --- DATA AUGMENTATION ---
#         if self.mode == 'train':
#             image, mask = self.augment(image, mask)
#         # ---------------------------------
#
#         # Thêm chiều Channels (1, 512, 512)
#         image = np.expand_dims(image.copy(), axis=0)
#         mask = np.expand_dims(mask.copy(), axis=0)
#
#         # Chuyển sang Tensor
#         image_tensor = torch.from_numpy(image)
#         mask_tensor = torch.from_numpy(mask)
#
#         return image_tensor, mask_tensor

import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
import random
import albumentations as A


class TransAttUnetDataset(Dataset):
    def __init__(self, data_dir, split_file, mode='train', multiplier=5):
        """
        Data Augmentation Hạng Nặng bằng Albumentations.
        multiplier: Số lần nhân bản dữ liệu trong 1 Epoch (Mặc định x5)
        """
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")
        self.mode = mode

        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Không tìm thấy file split: {split_file}")

        with open(split_file, 'r') as f:
            splits = json.load(f)

        self.file_list = splits.get(mode, [])

        if not self.file_list:
            print(f"Cảnh báo: Tập dữ liệu '{mode}' rỗng!")
        else:
            # --- 1. NHÂN BẢN DỮ LIỆU CHỈ CHO TẬP TRAIN ---
            if self.mode == 'train':
                original_len = len(self.file_list)
                self.file_list = self.file_list * multiplier
                print(
                    f"Đã load tập '{mode}': {original_len} mẫu. (Ép xung x{multiplier} -> {len(self.file_list)} mẫu/Epoch)")
            else:
                print(f"Đã load tập '{mode}': {len(self.file_list)} mẫu.")

        # --- 2. KHAI BÁO PIPELINE ALBUMENTATIONS ---
        if self.mode == 'train':
            self.aug_pipeline = A.Compose([
                # Nhóm 1: Hình học cơ bản (Giống code cũ của bạn nhưng chạy nhanh hơn)
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),

                # Nhóm 2: Biến dạng không gian (Mô phỏng nhịp thở, ép khối u méo mó)
                # interpolation=0 (cv2.INTER_NEAREST) để mask (0, 1, 2) không bị nội suy ra số thập phân
                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5, interpolation=0),
                    A.GridDistortion(p=0.5, interpolation=0),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5, interpolation=0),
                ], p=0.8),  # 80% cơ hội sẽ bị biến dạng

                # Nhóm 3: Nhiễu và bộ lọc (Giúp mô hình chai lỳ với máy chụp dởm)
                A.OneOf([
                    A.GaussNoise(var_limit=(0.001, 0.005), p=0.5),
                    A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                ], p=0.5)
            ])

    def __len__(self):
        return len(self.file_list)

    # ĐÃ XÓA HÀM `augment` BẰNG NUMPY CŨ ĐỂ DÙNG ALBUMENTATIONS

    def __getitem__(self, idx):
        fname = self.file_list[idx]

        img_path = os.path.join(self.image_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)

        # Load file .npy
        # Image shape: (512, 512)
        image = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        # --- 3. ÁP DỤNG DATA AUGMENTATION ---
        if self.mode == 'train':
            # Albumentations tự động xử lý cả ảnh và mask cùng lúc để khớp tọa độ
            augmented = self.aug_pipeline(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        # ---------------------------------

        # Thêm chiều Channels (1, 512, 512)
        # Lệnh .copy() rất quan trọng vì Albumentations có thể trả về negative strides
        image = np.expand_dims(image.copy(), axis=0)
        mask = np.expand_dims(mask.copy(), axis=0)

        # Chuyển sang Tensor
        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask)

        return image_tensor, mask_tensor