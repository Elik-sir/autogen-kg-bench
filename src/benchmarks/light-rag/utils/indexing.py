from __future__ import annotations

import settings  # noqa: WPS433 — пакет настроек каталога light-rag


async def require_indexing_complete(rag: object) -> str | None:
    """Если индексация не дошла до успешного статуса, вернуть сообщение об ошибке."""
    from lightrag.base import DocStatus  # noqa: WPS433

    failed = await rag.doc_status.get_docs_by_status(DocStatus.FAILED)
    if failed:
        lines = [
            f"  {doc_id}: {(st.error_msg or str(st.status)).strip()}"
            for doc_id, st in failed.items()
        ]
        return (
            f"Индексация не завершена: {len(failed)} документ(ов) в статусе FAILED "
            f"(см. LIGHTRAG_LLM_TIMEOUT_SEC в settings.py, сейчас {settings.LLM_TIMEOUT_SEC}s).\n"
            + "\n".join(lines)
        )

    incomplete = await rag.doc_status.get_docs_by_statuses(
        [DocStatus.PENDING, DocStatus.PROCESSING]
    )
    if incomplete:
        lines = [f"  {doc_id}: {st.status}" for doc_id, st in incomplete.items()]
        return (
            "Индексация не завершена: остались документы PENDING/PROCESSING:\n"
            + "\n".join(lines)
        )
    return None
