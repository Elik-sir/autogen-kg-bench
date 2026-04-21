import os
from openai import OpenAI

class LLMClient:
    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")

    def generate_response(self, system_prompt, user_prompt):
        """Отправляет запрос к LLM через OpenRouter."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2, 
            )
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Ошибка при запросе к LLM (OpenRouter): {e}")
            return None