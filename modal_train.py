import modal
import os
import subprocess

ignore_patterns = [
    ".venv", ".idea", "__pycache__", "data", ".idea", "processed.tar.gz", "venv"
]
# 1. Khởi tạo App
app = modal.App("transattunet-lidc-training")

# 2. Cấu hình môi trường chạy
REMOTE_ROOT = "/root/project"
image = (
    # 1. Image nền với CUDA 12.1 và NVCC
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")

    # 2. Cài tool cơ bản, BẮT BUỘC có g++
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "git", "build-essential", "g++", "gcc")

    # 3. Cài các dependency quan trọng trước tiên
    .pip_install("numpy", "packaging", "ninja", "wheel")

    # 4. Ép buộc cài đúng PyTorch 2.4.0 + cu121
    .pip_install("torch==2.4.0", "torchvision==0.19.0", index_url="https://download.pytorch.org/whl/cu121")

    # 5. CHÌA KHÓA: Ép biến môi trường CC và CXX trỏ thẳng vào gcc và g++
    .env({"CC": "gcc", "CXX": "g++"})

    # 6. Cài causal-conv1d và mamba-ssm (cờ --no-build-isolation vẫn giữ nguyên)
    # .run_commands(
    #     "pip install causal-conv1d>=1.2.0 --no-build-isolation",
    #     "pip install mamba-ssm --no-build-isolation"
    # )

    # 7. Các package còn lại
    .pip_install("pyyaml", "tqdm", "scikit-learn", "scipy", "git+https://github.com/lucasb-eyer/pydensecrf.git")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("pydensecrf")

    .add_local_dir(".", remote_path=REMOTE_ROOT, ignore=ignore_patterns, copy=True)
)

volume = modal.Volume.from_name("storage", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    volumes={"/mnt/storage": volume},
    timeout=86400
)

def train_remote(resume_path: str = None):
    import subprocess
    import os
    import sys

    env = os.environ.copy()
    env["PYTHONPATH"] = REMOTE_ROOT
    os.chdir(REMOTE_ROOT)

    log_dir = "/mnt/storage/logs"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "train.log")

    print(f"📄 Log sẽ lưu tại: {log_file}")

    cmd = [
        "python", "train.py",
        "--config", "configs/config.yaml"
    ]

    if resume_path:
        cmd.extend(["--resume", resume_path])

    with open(log_file, "a") as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env
        )

        for line in process.stdout:
            print(line, end="")   # vẫn hiện trên console
            f.write(line)         # ghi vào file
            f.flush()
            if "Epoch" in line or "Saved" in line:
                volume.commit()
        process.wait()
        volume.commit()

# ==================== INFERENCE / EVALUATION ====================
@app.function(
    image=image,
    gpu="A100",
    volumes={"/mnt/storage": volume},  # Volume được gắn ở /mnt/storage
    timeout=7200,
)
def evaluate_remote(
        model_path: str,
        vis_num: int = 50,
        config_path: str = "configs/config.yaml"
):
    import os
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = REMOTE_ROOT
    os.chdir(REMOTE_ROOT)

    # Đường dẫn lưu trên Volume (Ổ cứng bền vững)
    volume_output_dir = "/mnt/storage/results_roi"
    os.makedirs(volume_output_dir, exist_ok=True)

    print(f"Đang chạy Evaluation... Kết quả sẽ lưu tại: {volume_output_dir}")

    cmd = [
        "python", "inference.py",
        "--config", config_path,
        "--model_path", model_path,
        "--vis_num", str(vis_num),
        "--save_dir", volume_output_dir
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in process.stdout:
        print(line, end="")
    process.wait()

    print("\n✅ Đã chạy xong! Dữ liệu đã an toàn trên Volume.")

# 3. Hàm main tại Local để nhận tham số dòng lệnh
@app.local_entrypoint()
def main(resume: str = None):
    """
    Cách chạy:
    1. Train mới: modal run -d modal_train.py::main
    2. Stop : modal app list -> app ID -> modal app stop ap-hrd8ABR1bjJFFwEk5d2GzC
    3. Resume:    modal run -d modal_train.py::main --resume /mnt/storage/outputs/checkpoints/last_checkpoint.pth

    Load volume
    tạo volume 
    modal volume create storage
    # Upload folder data đã xử lý lên Volume
    modal volume put storage "\data\processed" /data/processed
    """
    train_remote.remote(resume_path=resume)

# ==================== LOCAL ENTRYPOINTS ====================
@app.local_entrypoint()
def train(resume: str = None):
    """Chạy training"""
    train_remote.remote(resume_path=resume)


@app.local_entrypoint()
def evaluate(
    model_path: str = modal.parameter(),
    vis_num: int = 50
):
    """
    Chạy evaluation/inference trên tập test.

    Ví dụ sử dụng:
    modal run -d modal_train.py::evaluate --model-path /mnt/storage/outputs/checkpoints/best_model.pth --vis-num 100
    """
    evaluate_remote.remote(
        model_path=model_path,
        vis_num=vis_num,
        config_path="configs/config.yaml"
    )