import modal

app = modal.App("upload-to-storage")  # Tên app tùy ý

# Lấy volume tên "storage" (như lệnh của bạn)
volume = modal.Volume.from_name("storage", create_if_missing=True)


@app.local_entrypoint()
def main():
    from pathlib import Path

    # Tự động lấy theo vị trí file upload_volume.py
    local_path = Path(__file__).parent / "data" / "processed"
    remote_path = "/data/processed"

    print(f"Đường dẫn local: {local_path.absolute()}")
    print(f"Thư mục tồn tại? {local_path.is_dir()}")

    if not local_path.is_dir():
        print("   Không tìm thấy thư mục data/processed!")
        print("   Kiểm tra xem trong project có folder 'data/processed' không.")
        return

    print(f"Đang upload {local_path} → {remote_path} (batch mode)...")

    with volume.batch_upload(force=True) as batch:
        batch.put_directory(str(local_path), remote_path)

    print("Upload hoàn tất!")