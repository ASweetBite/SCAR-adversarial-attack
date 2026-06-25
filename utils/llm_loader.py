# import requests
# import concurrent.futures
#
#
# class LocalLLMClient:
#     def __init__(self, model_name="qwen2.5-1.5b-code", host="http://localhost:11434"):#, host="http://localhost:11434"
#         self.model_name = model_name
#         self.host = host
#         print(f"[*] Initializing local LLM generator via Ollama ({self.model_name}) on {self.host}...")
#
#         try:
#             resp = requests.get(f"{self.host}/api/tags", timeout=5)
#             resp.raise_for_status()
#             models = [m['name'] for m in resp.json().get('models', [])]
#             if self.model_name not in models:
#                 print(
#                     f"[!] Warning: Model '{self.model_name}' not found. Please run `OLLAMA_HOST=127.0.0.1:11435 ollama pull {self.model_name}`.")
#             else:
#                 print(f"[*] Ollama connection successful. Model '{self.model_name}' is ready.")
#         except Exception as e:
#             print(f"[!] CRITICAL: Error connecting to Ollama at {self.host}. Error: {e}")
#
#     def chat(self, prompt: str) -> str:
#         system_prompt = "You are an expert C/C++ developer. Follow the user's instructions carefully. You can provide reasoning first, and then accurately output the JSON block."
#         payload = {
#             "model": self.model_name,
#             "messages": [
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": prompt}
#             ],
#             "stream": False,
#             "options": {
#                 "temperature": 0.85,
#                 "top_p": 0.95,
#                 "num_predict": 4096
#             }
#         }
#
#         try:
#             response = requests.post(f"{self.host}/api/chat", json=payload, timeout=300)
#             response.raise_for_status()
#             return response.json()['message']['content'].strip()
#         except Exception as e:
#             # 增强错误打印，避免 e.response 不存在时报错
#             if hasattr(e, 'response') and e.response is not None:
#                 print(f"[!] Ollama HTTP Error {e.response.status_code}: {e.response.text}")
#             else:
#                 print(f"[!] Ollama Request Failed: {e}")
#             return ""
#
#     def batch_chat(self, prompts: list[str]) -> list[str]:
#         if not prompts:
#             return []
#
#         # ⚠️ 注意这里：最大并发数对齐我们刚才设置的 OLLAMA_NUM_PARALLEL=6
#         max_workers = min(6, len(prompts))
#
#         responses = []
#         with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#             results = executor.map(self.chat, prompts)
#             responses = list(results)
#
#         return responses

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LocalLLMClient:
    def __init__(self, model_name="models/qwen2.5-1.5b-code", host=None):
        self.model_name = model_name
        print(f"[*] Initializing local HuggingFace LLM ({self.model_name}) directly...")

        try:
            # 1. 加载 Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            # ⚠️ 关键点：对于 Decoder-only 模型（如 Qwen），做 Batch 生成时必须使用左侧填充（left padding）
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 2. 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype="auto",  # 自动选择精度 (fp16/bf16/fp32)
                trust_remote_code=True
            )
            self.model.eval()

            print(f"[*] Local model '{self.model_name}' loaded successfully on {self.model.device}.")
        except Exception as e:
            print(f"[!] CRITICAL: Error loading HuggingFace model at {self.model_name}. Error: {e}")
            raise e

    def chat(self, prompt: str) -> str:
        # 单条生成的接口（复用 batch_chat 逻辑，保持代码整洁）
        return self.batch_chat([prompt])[0]

    def batch_chat(self, prompts: list[str], batch_size: int = 512) -> list[str]:
        """
        利用大显存进行真正的 Tensor 并发批处理
        batch_size: 每次丢给显卡同时计算的条数。显存越大，这个数字可以设置得越高（如 16, 32）。
        """
        if not prompts:
            return []

        system_prompt = "You are an expert C/C++ developer. Follow the user's instructions carefully. You can provide reasoning first, and then accurately output the JSON block."

        all_responses = []

        # 将所有的 prompts 按照 batch_size 切块
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i: i + batch_size]

            # 1. 构造标准对话格式
            messages_batch = [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": p}
                ] for p in batch_prompts
            ]

            # 2. 应用模板转换为纯文本列表
            text_batch = [
                self.tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True
                ) for msgs in messages_batch
            ]

            # 3. 对这批文本进行 Tokenize 并进行填充（Padding），转移到 GPU
            inputs = self.tokenizer(
                text_batch,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.model.device)

            # 4. 在 GPU 上同时生成这批回答
            try:
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=400,
                        temperature=0.85,
                        top_p=0.95,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )

                # 5. 切片分离：只保留模型新生成的部分（剥离掉输入的 prompt）
                input_length = inputs.input_ids.shape[1]
                generated_tokens = generated_ids[:, input_length:]

                # 6. 解码为文本
                batch_responses = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

                # 清理首尾空格并保存
                all_responses.extend([resp.strip() for resp in batch_responses])

            except Exception as e:
                print(f"[!] Local batch generation failed: {e}")
                # 如果某一批次失败（比如爆显存），用空字符串填补以保证长度一致
                all_responses.extend([""] * len(batch_prompts))

        return all_responses