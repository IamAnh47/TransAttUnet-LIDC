import os
import yaml
import numpy as np
import scipy.ndimage
import json
import random
import glob
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from pylidc.utils import consensus
import pylidc as pl
import uuid

# Import loader
from src.dicom_loader import DicomLoader


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def normalize_hu(image, window_center, window_width):
    """
    Áp dụng Lung Window và chuẩn hóa về [0, 1] dựa trên tham số truyền vào
    """
    wl = window_center
    ww = window_width

    upper, lower = wl + ww // 2, wl - ww // 2

    image = np.clip(image, lower, upper)
    image = (image - lower) / (upper - lower)
    return image.astype(np.float32)


# Phân đoạn phổi
def segment_lung_mask(volume):
    # Phân đoạn phổi dựa trên HU
    binary = volume < -320

    binary = scipy.ndimage.binary_opening(binary, structure=np.ones((3, 3, 3)))
    binary = scipy.ndimage.binary_closing(binary, structure=np.ones((5, 5, 5)))

    labels, num = scipy.ndimage.label(binary)
    if num == 0:
        return np.zeros_like(volume, dtype=np.uint8)

    sizes = scipy.ndimage.sum(binary, labels, range(1, num + 1))
    largest = np.argsort(sizes)[-2:] + 1

    mask = np.zeros_like(binary)
    for l in largest:
        mask[labels == l] = 1

    return mask.astype(np.uint8)


def process_patient_segmentation(args):
    # Bung nén tham số từ main truyền vào
    pid, raw_dir, processed_dir, seg_params = args

    loader = DicomLoader(raw_dir)

    img_dir = os.path.join(processed_dir, "images")
    mask_dir = os.path.join(processed_dir, "masks")

    stats = {"pid": pid, "slices": 0, "nodules": 0, "success": False, "error": None}

    try:
        vol, spacing, nodules = loader.load_patient_data(pid)
        if vol is None:
            stats["error"] = "Load Failed"
            return stats

        lung_mask = segment_lung_mask(vol)

        # Sử dụng tham số từ config
        vol_norm = normalize_hu(vol, seg_params['window_center'], seg_params['window_width'])

        valid_nodules = 0

        for i, cluster in enumerate(nodules):
            if len(cluster) < 3: continue

            # Convert padding list từ yaml sang tuple nếu cần thiết cho pylidc
            pad_val = seg_params['padding']
            # Đảm bảo format đúng cho pylidc [(x,x), (y,y), (z,z)]
            if isinstance(pad_val[0], list):
                pad_val = [tuple(p) for p in pad_val]

            mask_roi, cbbox, _ = consensus(cluster, clevel=seg_params['consensus_level'], pad=pad_val)

            z_start = cbbox[2].start
            z_stop = cbbox[2].stop

            for z in range(z_start, z_stop):
                slice_img = vol_norm[z, :, :]

                # multi-class mask
                # 0 = background
                # 1 = lung
                # 2 = nodule
                slice_mask = np.zeros_like(slice_img, dtype=np.uint8)

                slice_mask[lung_mask[z] > 0] = 1

                z_local = z - z_start
                mask_slice_roi = mask_roi[:, :, z_local]

                x_start, x_stop = cbbox[0].start, cbbox[0].stop
                y_start, y_stop = cbbox[1].start, cbbox[1].stop

                h_img, w_img = slice_img.shape

                x_s = max(0, x_start)
                x_e = min(h_img, x_stop)
                y_s = max(0, y_start)
                y_e = min(w_img, y_stop)

                if x_s >= x_e or y_s >= y_e: continue

                roi_x_s = x_s - x_start
                roi_x_e = roi_x_s + (x_e - x_s)
                roi_y_s = y_s - y_start
                roi_y_e = roi_y_s + (y_e - y_s)

                roi = mask_slice_roi[roi_x_s:roi_x_e, roi_y_s:roi_y_e]

                if roi.shape == (x_e - x_s, y_e - y_s):
                    temp_view = slice_mask[x_s:x_e, y_s:y_e]
                    temp_view[roi > 0] = 2

                    # --- CẮT PATCH 128x128 TẠI TÂM KHỐI U ---
                    # Chỉ cắt và lưu những slice THỰC SỰ CÓ KHỐI U
                if np.sum(slice_mask == 2) > 0:
                    patch_size = 128
                    half_patch = patch_size // 2

                    # 1. Tìm tâm của khối u trên slice này
                    cx = (x_s + x_e) // 2
                    cy = (y_s + y_e) // 2

                    # 2. Tính tọa độ khung cắt
                    x1, x2 = cx - half_patch, cx + half_patch
                    y1, y2 = cy - half_patch, cy + half_patch

                    # 3. Chống tràn viền (Nếu u nằm sát mép ảnh 512, đẩy khung cắt lùi lại)
                    if x1 < 0:
                        x2 += (0 - x1)
                        x1 = 0
                    elif x2 > h_img:
                        x1 -= (x2 - h_img)
                        x2 = h_img

                    if y1 < 0:
                        y2 += (0 - y1)
                        y1 = 0
                    elif y2 > w_img:
                        y1 -= (y2 - w_img)
                        y2 = w_img

                    # 4. Cắt ảnh và mask
                    patch_img = slice_img[x1:x2, y1:y2]
                    patch_mask = slice_mask[x1:x2, y1:y2]

                    # 5. Lưu lại nếu kích thước chuẩn 128x128
                    if patch_img.shape == (patch_size, patch_size):
                        file_id = f"{pid}_nodule{i}_slice{z}_{uuid.uuid4().hex[:6]}"
                        np.save(os.path.join(img_dir, f"{file_id}.npy"), patch_img)
                        np.save(os.path.join(mask_dir, f"{file_id}.npy"), patch_mask)

                        if "slices" not in stats:
                            stats["slices"] = 0
                        stats["slices"] += 1

            valid_nodules += 1

        if valid_nodules > 0:
            stats["success"] = True
            stats["nodules"] = valid_nodules
        else:
            stats["error"] = "No valid nodules"

    except Exception as e:
        import traceback
        stats["error"] = traceback.format_exc()

    return stats


def main():
    parser = argparse.ArgumentParser(description="Preprocess LIDC-IDRI")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--num", type=int, default=None, help="Số lượng bệnh nhân muốn xử lý (để trống sẽ chạy hết)")
    args = parser.parse_args()

    # 1. LOAD CONFIG
    print(f"Loading config from: {args.config}")
    cfg = load_config(args.config)

    # Lấy đường dẫn từ config
    RAW_DIR = cfg['paths']['raw_data']
    PROCESSED_DIR = cfg['paths']['processed_data']

    # Lấy tham số preprocessing từ config để truyền vào worker
    seg_params = cfg['data']

    os.makedirs(os.path.join(PROCESSED_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DIR, "masks"), exist_ok=True)

    # 2. LOAD & FILTER PATIENTS (RESUME LOGIC)
    loader = DicomLoader(RAW_DIR)
    all_patients = loader.get_all_patient_ids()

    processed_log = os.path.join(PROCESSED_DIR, "processed_pids.json")
    done_pids = set()

    if os.path.exists(processed_log):
        with open(processed_log, 'r') as f:
            done_pids = set(json.load(f))
        print(f" Resume: Đã tìm thấy {len(done_pids)} bệnh nhân đã xử lý trong log.")

    # Lọc bỏ những người đã làm rồi
    target_pids = [p for p in all_patients if p not in done_pids]

    # Áp dụng giới hạn số lượng (--num)
    if args.num is not None:
        print(f" Chế độ giới hạn: Chỉ xử lý {args.num} bệnh nhân tiếp theo.")
        target_pids = target_pids[:args.num]

    if not target_pids:
        print(" Không còn bệnh nhân nào cần xử lý hoặc tất cả đã hoàn thành.")
        # Vẫn chạy phần split data bên dưới phòng trường hợp muốn split lại
    else:
        print(f" Bắt đầu xử lý {len(target_pids)} bệnh nhân...")

        # Chạy song song
        max_workers = min(os.cpu_count(), 8)
        # Truyền seg_params vào mỗi task
        tasks = [(pid, RAW_DIR, PROCESSED_DIR, seg_params) for pid in target_pids]

        total_slices = 0
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_patient_segmentation, t) for t in tasks]

            for future in tqdm(as_completed(futures), total=len(tasks)):
                res = future.result()
                if res["success"]:
                    total_slices += res["slices"]
                    done_pids.add(res["pid"])
                else:
                    print(f"\n[LỖI] Bệnh nhân {res['pid']} thất bại vì: {res.get('error')}")

                # Lưu log định kỳ (safe write)
                if len(done_pids) % 10 == 0:
                    tmp_path = processed_log + ".tmp"
                    with open(tmp_path, "w") as f:
                        json.dump(list(done_pids), f)
                    os.replace(tmp_path, processed_log)

        # Lưu log cuối cùng
        with open(processed_log, "w") as f:
            json.dump(list(done_pids), f)
        print(f" Hoàn tất Preprocessing! Tổng số slices mới thu được: {total_slices}")

    # --- 3. CHIA TẬP DATASET (SPLIT) ---
    print("\n Đang kiểm tra và chia tập dữ liệu (Split)...")
    all_images = glob.glob(os.path.join(PROCESSED_DIR, "images", "*.npy"))

    if not all_images:
        print("Chưa có dữ liệu ảnh nào trong folder processed để split.")
        return

    patient_map = {}
    for fpath in all_images:
        fname = os.path.basename(fpath)
        pid = fname.split("_")[0]
        if pid not in patient_map: patient_map[pid] = []
        patient_map[pid].append(fname)

    pids = list(patient_map.keys())
    # Sắp xếp trước khi shuffle để đảm bảo kết quả giống nhau nếu cùng seed
    pids.sort()
    random.seed(42)
    random.shuffle(pids)

    n = len(pids)
    n_train = int(n * 0.7)
    n_val = int(n * 0.2)
    n_test = int(n * 0.1)

    if n > 0 and n_train == 0: n_train = n

    train_pids = pids[:n_train]
    val_pids = pids[n_train:n_train + n_val]
    test_pids = pids[n_train + n_val:]

    def get_files(pid_list):
        files = []
        for pid in pid_list: files.extend(patient_map[pid])
        return files

    split_data = {
        "train": get_files(train_pids),
        "val": get_files(val_pids),
        "test": get_files(test_pids)
    }

    split_path = cfg['paths']['split_file'] if 'split_file' in cfg['paths'] else os.path.join(PROCESSED_DIR,
                                                                                              "split.json")

    with open(split_path, "w") as f:
        json.dump(split_data, f, indent=4)

    print(f" Split xong và lưu tại: {split_path}")
    print(f"   Train: {len(split_data['train'])} files ({len(train_pids)} patients)")
    print(f"   Val:   {len(split_data['val'])} files ({len(val_pids)} patients)")
    print(f"   Test:  {len(split_data['test'])} files ({len(test_pids)} patients)")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()