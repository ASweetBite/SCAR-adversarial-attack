import os
import logging
import numpy as np
from typing import List, Tuple, Dict, Union

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

# ==========================================
# 1. CodeBERT Dual-Head Wrapper
# ==========================================
class CodeBERTDualHeadWrapper(nn.Module):
    def __init__(self, base_model, hidden_size, num_cwe_classes):
        """Initializes the dual-head wrapper with detection and classification heads."""
        super().__init__()
        self.base_model = base_model
        self.detection_head = nn.Linear(hidden_size, 2)
        self.classification_head = nn.Linear(hidden_size, num_cwe_classes)

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, **kwargs):
        """Performs a forward pass and returns detection logits, classification logits, and hidden states."""
        out = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )

        cls_rep = out.last_hidden_state[:, 0, :]

        det_logits = self.detection_head(cls_rep)
        cls_logits = self.classification_head(cls_rep)

        return det_logits, cls_logits, out.hidden_states

    def get_input_embeddings(self):
        """Returns the input embeddings from the base model."""
        return self.base_model.get_input_embeddings()


# ==========================================
# 2. CodeBERT Specific Model Loader
# ==========================================
class CodeBERTModelLoader:
    def __init__(self, config):
        """Parses the configuration and sets up parameters for model loading."""
        self.config = config
        self.model_dir = config["model"]["model_name"]
        self.max_seq_len = config["model"]["max_seq_len"]

        if self.max_seq_len > 512:
            logger.warning(f"CodeBERT native max sequence length is 512, truncated max_seq_len from {self.max_seq_len} to 512.")
            self.max_seq_len = 512

        target_device = config["model"].get("device", "cuda:0")
        if torch.cuda.is_available():
            self.device = target_device if "cuda" in target_device else "cuda:0"
        else:
            self.device = "cpu"

        self.num_classes = config["data"].get("num_classes", 16)

    def load_model(self) -> Tuple["CodeBERTModel", AutoTokenizer]:
        """Loads the tokenizer, base model, and dual-head weights, placing them on the appropriate device."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        logger.info(f"Loading CodeBERT base model from: {self.model_dir}")
        base_model = AutoModel.from_pretrained(self.model_dir)

        hidden_size = base_model.config.hidden_size
        model = CodeBERTDualHeadWrapper(base_model, hidden_size, self.num_classes)

        head_path = os.path.join(self.model_dir, "dual_heads.pt")
        if os.path.exists(head_path):
            logger.info(f"Loading pretrained dual-head weights: {head_path}")
            head_state = torch.load(head_path, map_location="cpu")
            model.detection_head.load_state_dict(head_state["detection_head"])
            model.classification_head.load_state_dict(head_state["classification_head"])
            model.eval()
        else:
            logger.warning("No dual_heads.pt detected, initializing with random weights for training.")
            model.train()

        model.to(self.device)

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            logger.info(f"CodeBERT loaded successfully, current VRAM usage: {alloc:.2f} MB")

        return CodeBERTModel(
            model=model,
            tokenizer=tokenizer,
            max_seq_len=self.max_seq_len,
            device=self.device,
        ), tokenizer


# ==========================================
# 3. Encapsulated Inference Interface
# ==========================================
class CodeBERTModel:
    def __init__(self, model, tokenizer, max_seq_len, device):
        """Initializes the inference interface with the configured model and tokenizer."""
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.device = device

    @torch.no_grad()
    def predict(self, code: str) -> Dict[str, Union[float, List[float]]]:
        """Performs inference on a single code snippet to yield vulnerability and classification probabilities."""
        self.model.eval()
        inputs = self.tokenizer(
            code, truncation=True, padding="max_length",
            max_length=self.max_seq_len, return_tensors="pt",
        ).to(self.device)

        det_logits, cls_logits, _ = self.model(**inputs)

        det_probs = torch.softmax(det_logits, dim=1).float().cpu().numpy()[0]
        cls_probs = torch.softmax(cls_logits, dim=1).float().cpu().numpy()[0]

        return {
            "f_det": float(det_probs[1]),
            "f_cls": cls_probs.tolist(),
        }

    @torch.no_grad()
    def batch_predict(self, code_list: List[str], batch_size: int = 16) -> Dict[str, np.ndarray]:
        """Performs batched inference on a list of code snippets."""
        self.model.eval()
        all_det_probs = []
        all_cls_probs = []

        for i in range(0, len(code_list), batch_size):
            batch = code_list[i: i + batch_size]
            inputs = self.tokenizer(
                batch, truncation=True, padding="max_length",
                max_length=self.max_seq_len, return_tensors="pt",
            ).to(self.device)

            det_logits, cls_logits, _ = self.model(**inputs)

            det_probs = torch.softmax(det_logits, dim=1).float().cpu().numpy()
            cls_probs = torch.softmax(cls_logits, dim=1).float().cpu().numpy()

            all_det_probs.append(det_probs)
            all_cls_probs.append(cls_probs)

        if not all_det_probs:
            return {"f_det": np.array([]), "f_cls": np.array([])}

        return {
            "f_det": np.vstack(all_det_probs)[:, 1],
            "f_cls": np.vstack(all_cls_probs),
        }

    @torch.no_grad()
    def encode(self, code: str):
        """Generates and returns the hidden states of the given code snippet."""
        self.model.eval()
        inputs = self.tokenizer(
            code, truncation=True, padding="max_length",
            max_length=self.max_seq_len, return_tensors="pt",
        ).to(self.device)

        _, _, hidden_states = self.model(**inputs)
        return hidden_states