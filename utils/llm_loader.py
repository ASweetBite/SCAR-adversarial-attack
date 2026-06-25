import requests
import concurrent.futures

class LocalLLMClient:
    def __init__(self, model_name="qwen3.6:latest", host="http://localhost:11434"):
        self.model_name = model_name
        self.host = host
        print(f"[*] Initializing local LLM generator via Ollama ({self.model_name}) on {self.host}...")

        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m['name'] for m in resp.json().get('models', [])]
            if self.model_name not in models:
                print(
                    f"[!] Warning: Model '{self.model_name}' not found. Please run `OLLAMA_HOST=127.0.0.1:11435 ollama pull {self.model_name}`.")
            else:
                print(f"[*] Ollama connection successful. Model '{self.model_name}' is ready.")
        except Exception as e:
            print(f"[!] CRITICAL: Error connecting to Ollama at {self.host}. Error: {e}")

    def chat(self, prompt: str) -> str:
        system_prompt = "You are a precise C/C++ refactoring assistant. Always output valid JSON exactly as requested by the user. Do not include markdown code blocks, conversational text, or explanations."

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.85,
                "top_p": 0.95,
                "num_predict": 1500,
                "num_ctx": 8192
            }
        }

        try:
            response = requests.post(f"{self.host}/api/chat", json=payload, timeout=300)
            response.raise_for_status()
            return response.json()['message']['content'].strip()
        except requests.exceptions.RequestException as e:
            error_msg = e.response.text if getattr(e, 'response', None) is not None else str(e)
            print(f"[!] Ollama HTTP Error: {error_msg}")
            return ""

    def batch_chat(self, prompts: list[str]) -> list[str]:
        if not prompts:
            return []

        # ⚠️ 注意这里：最大并发数对齐 Ollama 服务端设置的并发能力
        max_workers = min(6, len(prompts))

        responses = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # map 保证了返回的 responses 顺序与输入的 prompts 顺序严格一一对应
            results = executor.map(self.chat, prompts)
            responses = list(results)

        return responses

# import json
# import re
# from typing import Dict, Any, List
#
# import torch
# import torch.nn.functional as F
# from transformers import AutoTokenizer, AutoModelForCausalLM
#
# class LocalLLMClient:
#     def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
#         # Initializes the local LLM client.
#         # [!] OPTIMIZATION: 移除 4-bit 量化，使用原生 bfloat16，充分释放 80GB 显存算力提升推理速度
#         print(f"[*] Initializing local LLM generator ({model_name}) in native BF16...")
#
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#         self.tokenizer.padding_side = 'left'
#
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
#
#         # 启用 FlashAttention 兼容模式
#         attn_impl = "sdpa"
#
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             device_map="auto",
#             torch_dtype=torch.bfloat16,  # 80GB 显卡完美支持 BF16，无精度损失且极速
#             trust_remote_code=True,
#             attn_implementation=attn_impl
#         )
#         self.model.eval()
#
#         self.model.config.pad_token_id = self.tokenizer.pad_token_id
#
#     @torch.no_grad()
#     def chat(self, prompt: str) -> str:
#         # Performs a single chat turn with low latency optimization.
#         messages = [
#             {"role": "system", "content": "You are a precise coding assistant. Output ONLY a comma-separated list of alternative variable names. No explanations."},
#             {"role": "user", "content": prompt}
#         ]
#         text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
#
#         outputs = self.model.generate(
#             **inputs,
#             max_new_tokens=256,
#             temperature=0.6,
#             top_p=0.9,
#             do_sample=True,
#             pad_token_id=self.tokenizer.pad_token_id,
#             eos_token_id=self.tokenizer.eos_token_id
#         )
#
#         input_len = inputs.input_ids.shape[1]
#         response = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
#         return response
#
#     @torch.no_grad()
#     def batch_chat(self, prompts: list[str], batch_size: int = 64) -> list[str]:
#         # Performs batch chat inference utilizing parallel processing.
#         # [!] OPTIMIZATION: 修正批处理并发度，增加动态截断与显存回收，防止极端情况的 OOM
#         if not prompts:
#             return []
#
#         texts = []
#         for prompt in prompts:
#             messages = [
#                 {"role": "system",
#                  "content": "You are a precise coding assistant. Output ONLY a comma-separated list of alternative variable names. No explanations."},
#                 {"role": "user", "content": prompt}
#             ]
#             texts.append(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
#
#         responses = []
#         for i in range(0, len(texts), batch_size):
#             batch_texts = texts[i: i + batch_size]
#
#             # 增加 truncation 和 max_length 保护，避免代码过长导致的 OOM 或张量报错
#             inputs = self.tokenizer(
#                 batch_texts,
#                 return_tensors="pt",
#                 padding=True,
#                 truncation=True,
#                 max_length=4096
#             ).to(self.model.device)
#
#             try:
#                 outputs = self.model.generate(
#                     **inputs,
#                     max_new_tokens=400,
#                     temperature=0.85,
#                     top_p=0.95,
#                     do_sample=True,
#                     pad_token_id=self.tokenizer.pad_token_id,
#                     eos_token_id=self.tokenizer.eos_token_id
#                 )
#
#                 input_len = inputs.input_ids.shape[1]
#                 for output in outputs:
#                     responses.append(self.tokenizer.decode(output[input_len:], skip_special_tokens=True).strip())
#
#             except Exception as e:
#                 # 如果某个 Batch 崩溃（例如依然超显存），输出具体异常并对齐返回列表长度，避免后续解包错位
#                 print(f"\n[!] Sub-batch generation failed: {e}")
#                 responses.extend([""] * len(batch_texts))
#
#             finally:
#                 # 显式清理缓存，防止不同长度 batch 切换时的显存碎片化
#                 del inputs
#                 if 'outputs' in locals():
#                     del outputs
#                 torch.cuda.empty_cache()
#
#         return responses