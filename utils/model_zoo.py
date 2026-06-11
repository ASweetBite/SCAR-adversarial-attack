import logging
import os
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


class ModelZooQueryTracker:
    def __init__(self, model_zoo):
        # Initializes the query tracker by wrapping a model zoo instance.
        self._model_zoo = model_zoo
        self._query_count = 0

    def reset_counter(self):
        # Resets the query counter to zero.
        self._query_count = 0

    def get_query_count(self):
        # Returns the total number of intercepted queries.
        return self._query_count

    def predict(self, *args, **kwargs):
        # Intercepts a single prediction query and increments the counter.
        self._query_count += 1
        return self._model_zoo.predict(*args, **kwargs)

    def batch_predict(self, codes, *args, **kwargs):
        # Intercepts a batch prediction query and increments the counter by batch size.
        self._query_count += len(codes)
        return self._model_zoo.batch_predict(codes, *args, **kwargs)

    def predict_label_conf(self, *args, **kwargs):
        # Intercepts a label confidence query and increments the counter.
        self._query_count += 1
        return self._model_zoo.predict_label_conf(*args, **kwargs)

    def __getattr__(self, name):
        # Transparently forwards attribute lookups to the underlying model zoo.
        return getattr(self._model_zoo, name)


class ModelZoo:
    def __init__(self, model_configs: dict, eval_mode: str, config: dict):
        # Initializes the ModelZoo by loading classification models.
        glob_cfg = config.get('global', {})
        run_cfg = config.get('run_params', {})

        self.device = torch.device(glob_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.eval_mode = eval_mode
        self.num_classes = run_cfg.get('num_classes', 16)
        self.max_seq_len = run_cfg.get('max_seq_len', 512)

        self.models = {}
        self.model_names = list(model_configs.keys())

        for name, path in model_configs.items():
            print(f"[*] Loading Model[{name}] from {path}...")
            if not os.path.exists(path):
                print(f"[!] Path {path} not found. Skipping {name}.")
                continue

            try:
                print(f"[*] Loading standard HF classifier...")
                tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
                model = AutoModelForSequenceClassification.from_pretrained(path, trust_remote_code=True).to(
                    self.device)
                model.eval()
                self.models[name] = {"type": "standard", "tokenizer": tokenizer, "model": model}
            except Exception as e:
                print(f"[!] Error loading {name}: {e}")

    def predict(self, code: str, target_model: str) -> Tuple[List[float], int]:
        # Predicts the label for a single code snippet using the target model.
        m = self.models.get(target_model)
        if m is None:
            return [1.0, 0.0], -1

        inputs = m["tokenizer"](
            code, return_tensors="pt", truncation=True, max_length=self.max_seq_len, padding="max_length"
        ).to(self.device)

        with torch.no_grad():
            outputs = m["model"](**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy().tolist()
            pred_label = int(np.argmax(probs))

            if self.eval_mode == "binary" and pred_label == 0:
                pred_label = -1

        return probs, pred_label

    def batch_predict(self, codes: List[str], target_model: str, batch_size: int = 32) -> Tuple[
        List[List[float]], List[int]]:
        # Predicts labels for a batch of code snippets using the target model.
        m = self.models.get(target_model)
        if m is None:
            return [[1.0, 0.0]] * len(codes), [-1] * len(codes)

        all_probs, all_preds = [], []
        for i in range(0, len(codes), batch_size):
            batch_codes = codes[i:i + batch_size]
            inputs = m["tokenizer"](
                batch_codes, return_tensors="pt", truncation=True, max_length=self.max_seq_len, padding="max_length"
            ).to(self.device)

            with torch.no_grad():
                outputs = m["model"](**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                probs = [probs.tolist()] if probs.ndim == 1 else probs.tolist()
                preds = [int(np.argmax(p)) for p in probs]

                if self.eval_mode == "binary":
                    preds = [1 if p == 1 else -1 for p in preds]

                all_probs.extend(probs)
                all_preds.extend(preds)
        return all_probs, all_preds

    def predict_label_conf(self, code: str, label: int, target_model: str) -> float:
        # Retrieves the confidence score for a specific label of a given code.
        probs, _ = self.predict(code, target_model)
        return probs[label] if label < len(probs) else 0.0