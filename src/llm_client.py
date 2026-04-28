import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
class LLMClient:
    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = os.getenv("LLM_MODEL")

    @staticmethod
    def _shrink_prompt(prompt: str, keep_ratio: float = 0.7) -> str:
        text = str(prompt or "")
        if not text:
            return text
        new_len = max(2000, int(len(text) * keep_ratio))
        if new_len >= len(text):
            return text
        head = int(new_len * 0.7)
        tail = new_len - head
        return (
            f"{text[:head]}\n"
            f"...[PROMPT COMPRESSED FROM {len(text)} TO {new_len} CHARS]...\n"
            f"{text[-tail:]}"
        )

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
            error_text = str(e)
            if "maximum context length" in error_text.lower():
                print("Контекст слишком большой, пробую сжать prompt и повторить запрос...")
                try:
                    shrunk_prompt = self._shrink_prompt(user_prompt, keep_ratio=0.6)
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": shrunk_prompt}
                        ],
                        temperature=0.2,
                    )
                    return response.choices[0].message.content
                except Exception as retry_error:
                    print(f"Ошибка при повторном запросе к LLM (OpenRouter): {retry_error}")
                    return None
            print(f"Ошибка при запросе к LLM (OpenRouter): {e}")
            return None