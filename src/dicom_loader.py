import os
import glob
import pydicom
import pylidc as pl
import numpy as np
import configparser

# Vá lỗi numpy cũ
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'bool'): np.bool = bool


class DicomLoader:
    def __init__(self, raw_data_dir):
        self.raw_data_dir = raw_data_dir
        self.path_cache = {}
        # Patch configparser cho pylidc cũ
        if not hasattr(configparser, 'SafeConfigParser'):
            configparser.SafeConfigParser = configparser.ConfigParser
        self._activate_smart_path_finding()

    def _activate_smart_path_finding(self):
        root_dir = self.raw_data_dir
        cache = self.path_cache

        def smart_path_method(scan_instance):
            current_id = scan_instance.patient_id
            target_series_uid = scan_instance.series_instance_uid
            cache_key = f"{current_id}_{target_series_uid}"

            if cache_key in cache: return cache[cache_key]

            patient_dir = os.path.join(root_dir, current_id)
            if not os.path.exists(patient_dir):
                # Fallback tìm đệ quy
                search = glob.glob(os.path.join(root_dir, "**", current_id, "**", "*.dcm"), recursive=True)
                if search: return os.path.dirname(search[0])
                return patient_dir

            for root, dirs, files in os.walk(patient_dir):
                if not any(f.endswith('.dcm') for f in files): continue
                try:
                    ds = pydicom.dcmread(os.path.join(root, files[0]), stop_before_pixels=True)
                    if ds.SeriesInstanceUID == target_series_uid:
                        cache[cache_key] = root
                        return root
                except:
                    continue
            return patient_dir

        pl.Scan.get_path_to_dicom_files = smart_path_method

    def get_all_patient_ids(self):
        # Lấy ID từ tên folder cấp 1
        return sorted([d for d in os.listdir(self.raw_data_dir)
                       if os.path.isdir(os.path.join(self.raw_data_dir, d)) and d.startswith("LIDC-IDRI")])

    def load_patient_data(self, patient_id):
        scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).first()
        if not scan: return None, None, None
        try:
            images = scan.load_all_dicom_images(verbose=False)
            if not images: return None, None, None
            vol = np.stack([s.pixel_array * s.RescaleSlope + s.RescaleIntercept for s in images]).astype(np.float32)
            spacing = (scan.slice_spacing, scan.pixel_spacing, scan.pixel_spacing)
            return vol, spacing, scan.cluster_annotations()
        except:
            return None, None, None