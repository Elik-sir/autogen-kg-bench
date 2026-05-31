import os
import re
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
        # Явно ограничиваем max_tokens, иначе у некоторых моделей OpenRouter
        # может выставлять очень высокий потолок по умолчанию.
        self.max_tokens = max(
            64,
            int(os.getenv("OPENROUTER_MAX_TOKENS", os.getenv("LLM_MAX_TOKENS", "1024"))),
        )

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

    @staticmethod
    def _extract_affordable_tokens(error_text: str) -> int | None:
        # Пример: "... but can only afford 4541."
        m = re.search(r"can only afford\s+(\d+)", str(error_text), flags=re.IGNORECASE)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    def _request(self, system_prompt: str, user_prompt: str, max_tokens: int):
        return self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=max(64, int(max_tokens)),
        )

    def generate_response(self, system_prompt, user_prompt):
        """Отправляет запрос к LLM через OpenRouter."""
        max_tokens = self.max_tokens
        prompt = user_prompt

        # До 3 попыток: базовая, сжатие по context length, снижение по бюджету (402).
        for _ in range(3):
            try:
                response = self._request(system_prompt, prompt, max_tokens=max_tokens)
                return response.choices[0].message.content
            except Exception as e:
                error_text = str(e)
                lowered = error_text.lower()

                if "maximum context length" in lowered:
                    print("Контекст слишком большой, пробую сжать prompt и повторить запрос...")
                    prompt = self._shrink_prompt(prompt, keep_ratio=0.6)
                    continue

                if "error code: 402" in lowered or "'code': 402" in lowered:
                    affordable = self._extract_affordable_tokens(error_text)
                    if affordable is None:
                        affordable = max(64, max_tokens // 2)
                    # Даем запас, чтобы точно влезть в бюджет текущего ключа.
                    max_tokens = max(64, min(max_tokens // 2, affordable - 64))
                    print(
                        f"Недостаточно кредитов/лимита токенов, пробую с меньшим max_tokens={max_tokens}..."
                    )
                    if max_tokens <= 64:
                        break
                    continue

                print(f"Ошибка при запросе к LLM (OpenRouter): {e}")
                return None

        print("Ошибка при запросе к LLM (OpenRouter): исчерпаны попытки повтора.")
        return None