# rag/ollama.py
import logging
import requests


class OllamaGenerator:
    """LLM Generator using locally hosted Ollama"""

    def __init__(self, config):
        self.config = config
        self.base_url = config.ollama_url
        self.model = config.llm_model

        # Test connection
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise RuntimeError("Ollama not responding")
        except Exception as e:
            logging.error(f"Failed to connect to Ollama: {e}")
            raise

        logging.info(f"Ollama connected: {self.model}")

    def generate(self, prompt: str) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.llm_temperature,
                        "num_predict": self.config.llm_max_tokens,
                    },
                },
                timeout=120,
            )

            if response.status_code != 200:
                return f"LLM error: {response.text}"

            return response.json().get("response", "").strip()

        except requests.exceptions.Timeout:
            return "Error: LLM request timed out"
        except Exception as e:
            return f"Error generating response: {e}"
