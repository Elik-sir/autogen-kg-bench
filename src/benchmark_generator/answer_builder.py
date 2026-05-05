from __future__ import annotations


def build_answer_from_context(llm, question: str, ground_truth: str, fallback: str = "") -> str:
    """
    Формирует эталонный answer по тому же принципу, что и subgraph-deep-analytics:
    LLM получает вопрос + контекст и возвращает краткий проверяемый ответ.
    """
    if not str(ground_truth or "").strip():
        return str(fallback or "").strip()
    system_prompt = (
        "You are an analyst who writes reference answers for a GraphRAG benchmark. "
        "Answer only from the given context-no invention or outside knowledge. "
        "If the context contains a direct answer, you must return it and must not claim insufficient data."
    )
    user_prompt = f"""
Question:
{question}

Context (ground_truth):
{ground_truth}

Produce a short, precise reference answer based only on this context.

MANDATORY RULES:
1) If the context contains a direct answer, you must return it.
2) Do not write "no information", "cannot determine", "insufficient data", or similar hedges
   when the context contains relevant entities.
3) Do not invent facts: use only what is explicitly present in the context.
4) For questions like "who works at which company", return pairs as "<employee>, <company>" (one pair per line).
5) If there is truly no relevant data, reply only with: "No data for this query."
6) No explanations or markdown-only the final answer text.
7) Write the entire answer in English (keep proper names and literals as in the context).
"""
    response = llm.generate_response(system_prompt, user_prompt)
    if response is None:
        return str(fallback or ground_truth).strip()
    answer = str(response).strip()
    return answer or str(fallback or ground_truth).strip()
