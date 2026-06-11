import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM


class MLMEngine:
    """Handles context-based code mask prediction and feature extraction using MLM."""

    def __init__(self, model_name="microsoft/codebert-base-mlm", local_dir="./models"):
        """Initializes the tokenizer and model on the available hardware device."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 构建本地模型保存路径，例如: ./models/microsoft/codebert-base-mlm
        self.model_path = os.path.join(local_dir, model_name)

        # 1. 判断本地是否存在该模型目录
        if os.path.exists(self.model_path):
            print(f"[*] 优先从本地加载 MLM 模型 ({self.model_path})...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_path).to(self.device)
        else:
            # 2. 如果本地不存在，则从 Hugging Face 下载
            print(f"[*] 本地未找到模型，正在从 Hugging Face 下载 ({model_name})...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)

            # 3. 下载完成后，将其保存到本地目录
            print(f"[*] 下载完成，正在保存到本地目录 ({self.model_path})...")
            os.makedirs(self.model_path, exist_ok=True)
            self.tokenizer.save_pretrained(self.model_path)
            self.model.save_pretrained(self.model_path)
            print("[*] 保存成功！下次运行将自动从本地读取。")

        self.model.eval()

    def get_embedding(self, code: str) -> np.ndarray:
        """Generates a high-dimensional embedding for a code snippet using the CLS token."""
        tokens = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=512,
                                padding="max_length").to(self.device)
        with torch.no_grad():
            outputs = self.model.base_model(**tokens)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embedding.squeeze()