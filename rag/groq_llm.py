import logging
from groq import Groq
from .config import Config

class GroqGenerator:
    def __init__(self, config: Config):
        if not config.groq_api_key:
            raise ValueError("GROQ_API_KEY not set")

        self.client = Groq(api_key=config.groq_api_key)
        self.model = config.llm_model
        self.temperature = config.llm_temperature
        self.max_tokens = config.llm_max_tokens

        logging.info(f"Groq LLM initialized → {self.model}")

    def generate(self, prompt: str) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Groq error: {e}")
            return "Error generating response from LLM."
