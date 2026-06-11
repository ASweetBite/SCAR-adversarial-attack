import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM


class MLMEngine:
    def __init__(self, model_name="microsoft/codebert-base-mlm", local_dir="./models"):
        # Initializes the tokenizer and model, loading from local path if available.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = os.path.join(local_dir, model_name)

        if os.path.exists(self.model_path):
            print(f"[*] Loading MLM model from local directory ({self.model_path})...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_path).to(self.device)
        else:
            print(f"[*] Model not found locally, downloading from Hugging Face ({model_name})...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)

            print(f"[*] Download complete, saving to local directory ({self.model_path})...")
            os.makedirs(self.model_path, exist_ok=True)
            self.tokenizer.save_pretrained(self.model_path)
            self.model.save_pretrained(self.model_path)
            print("[*] Save successful! Future runs will load from local.")

        self.model.eval()

    def get_embedding(self, code: str) -> np.ndarray:
        # Generates a high-dimensional embedding for a code snippet using the CLS token.
        tokens = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=512,
                                padding="max_length").to(self.device)
        with torch.no_grad():
            outputs = self.model.base_model(**tokens)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embedding.squeeze()
