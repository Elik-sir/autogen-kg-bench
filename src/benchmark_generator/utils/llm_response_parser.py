import json
import re


def parse_qa_pairs_response(response):
    if not response:
        return []

    cleaned_text = response.strip()
    if cleaned_text.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned_text, re.DOTALL | re.IGNORECASE)
        if match:
            cleaned_text = match.group(1).strip()

    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as error:
        print(f"Ошибка парсинга LLM ответа: {error}\nОтвет:\n{response}")
        return []
