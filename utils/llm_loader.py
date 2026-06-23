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
        system_prompt = "You are a precise coding assistant. Output ONLY a comma-separated list of alternative variable names. No explanations."

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
                "num_predict": 256
            }
        }

        try:
            response = requests.post(f"{self.host}/api/chat", json=payload, timeout=300)
            response.raise_for_status()
            return response.json()['message']['content'].strip()
        except Exception as e:
            # 这样就能清楚地看到 Ollama 到底在抱怨什么了
            print(f"[!] Ollama HTTP Error {e.response.status_code}: {e.response.text}")
            return ""

    def batch_chat(self, prompts: list[str]) -> list[str]:
        if not prompts:
            return []

        # ⚠️ 注意这里：最大并发数对齐我们刚才设置的 OLLAMA_NUM_PARALLEL=6
        max_workers = min(6, len(prompts))

        responses = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(self.chat, prompts)
            responses = list(results)

        return responses