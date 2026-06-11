import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class LocalLLMClient:
    """优化后的轻量级本地 LLM 客户端 - 已切换至 SDPA 模式"""

    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        print(f"[*] 正在初始化本地 LLM 生成器 ({model_name})...")

        # 4-bit 量化配置
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.padding_side = 'left'

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        attn_impl = "sdpa"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation=attn_impl  # 锁定为 sdpa
        )
        self.model.eval()

        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    @torch.no_grad()
    def chat(self, prompt: str) -> str:
        """单次对话（低延迟优化）"""
        messages = [
            {"role": "system", "content": "You are a precise coding assistant. Output ONLY a comma-separated list of alternative variable names. No explanations."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        input_len = inputs.input_ids.shape[1]
        response = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        return response

    @torch.no_grad()
    def batch_chat(self, prompts: list[str]) -> list[str]:
        """批量对话：利用 SDPA 的并行能力同时处理多个变量"""
        if not prompts:
            return []

        texts = []
        for prompt in prompts:
            messages = [
                {"role": "system", "content": "You are a precise coding assistant. Output ONLY a comma-separated list of alternative variable names. No explanations."},
                {"role": "user", "content": prompt}
            ]
            texts.append(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        # 批量编码，开启 padding
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.85,
            top_p=0.95,
            # repetition_penalty=1.15,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        responses = []
        input_len = inputs.input_ids.shape[1]
        for output in outputs:
            responses.append(self.tokenizer.decode(output[input_len:], skip_special_tokens=True).strip())

        return responses