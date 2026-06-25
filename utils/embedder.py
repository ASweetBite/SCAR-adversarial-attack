import os
import torch
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import snapshot_download


class CodeEmbedder:
    def __init__(self, model_name="microsoft/unixcoder-base", local_dir="./models", device="cuda"):
        self.device = device
        # 本地存储路径拼接
        self.model_path = os.path.join(local_dir, model_name)

        # 1. 检查本地环境，如果完全没有才去下载
        if not os.path.exists(self.model_path):
            print(f"[*] Model not found locally. Downloading full snapshot ({model_name})...")
            os.makedirs(self.model_path, exist_ok=True)
            snapshot_download(
                repo_id=model_name,
                local_dir=self.model_path,
                local_dir_use_symlinks=False
            )
            print("[*] Full download complete!")

        # 2. 纯粹的本地断网加载（无需任何 Hack，极其稳定）
        print(f"[*] Loading code embedding strictly from LOCAL directory ({self.model_path})...")

        # UniXcoder 是标准模型，直接用 local_files_only=True 完美离线
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True
        )

        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(device)[
            0] >= 8 else torch.float16

        self.model = AutoModel.from_pretrained(
            self.model_path,
            local_files_only=True,
            attn_implementation="sdpa",  # 依然享受极速推理
            torch_dtype=dtype
        ).to(self.device)

        self.model.eval()