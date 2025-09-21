import os, requests

class JetClient:
    def __init__(self, base_url: str, api_key: str = None, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("JET_API_KEY", "")
        self.timeout = timeout

    def chat_completions(self, model: str, messages, **kwargs):
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": messages}
        payload.update(kwargs)
        r = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()
