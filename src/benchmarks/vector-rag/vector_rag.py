"""
FAISS + OpenAI-совместимые эмбеддинги и LLM (LangChain).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import settings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _api_headers() -> dict[str, str] | None:
    if not settings.OPENROUTER_HTTP_REFERER:
        return None
    return {
        "HTTP-Referer": settings.OPENROUTER_HTTP_REFERER,
        "X-Title": settings.OPENROUTER_APP_TITLE,
    }


def get_embeddings() -> OpenAIEmbeddings:
    kw: dict[str, Any] = {
        "model": settings.EMBEDDING_MODEL,
        "api_key": settings.OPENROUTER_API_KEY,
        "openai_api_base": settings.OPENAI_API_BASE,
        "check_embedding_ctx_length": False,
    }
    h = _api_headers()
    if h:
        kw["default_headers"] = h
    return OpenAIEmbeddings(**kw)


def get_llm() -> ChatOpenAI:
    kw: dict[str, Any] = {
        "model": settings.LLM_MODEL,
        "api_key": settings.OPENROUTER_API_KEY,
        "openai_api_base": settings.OPENAI_API_BASE,
        "temperature": settings.LLM_TEMPERATURE,
    }
    h = _api_headers()
    if h:
        kw["default_headers"] = h
    return ChatOpenAI(**kw)


def _index_path(work: Path) -> Path:
    return work / "faiss_index"


def build_or_load_vectorstore(
    corpus_text: str,
    work: Path,
) -> FAISS:
    if not settings.OPENROUTER_API_KEY.strip():
        raise RuntimeError("Нужен OPENROUTER_API_KEY в settings.py или .env")
    work.mkdir(parents=True, exist_ok=True)
    emb = get_embeddings()
    index_dir = _index_path(work)
    if (
        not settings.REBUILD_INDEX
        and (index_dir / "index.faiss").is_file()
    ):
        return FAISS.load_local(
            str(index_dir),
            emb,
            allow_dangerous_deserialization=True,
        )
    if index_dir.exists():
        shutil.rmtree(index_dir)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(corpus_text)
    documents = [Document(page_content=c) for c in chunks if c.strip()]
    if not documents:
        raise ValueError("После нарезки чанков корпус пуст")
    store = FAISS.from_documents(documents, emb)
    store.save_local(str(index_dir))
    return store


def format_docs(docs: list) -> str:
    return "\n\n---\n\n".join(
        d.page_content if hasattr(d, "page_content") else str(d) for d in docs
    )


def answer_from_store(
    store: FAISS,
    question: str,
) -> str:
    retriever = store.as_retriever(search_kwargs={"k": settings.RETRIEVAL_K})
    docs = retriever.invoke(question)
    context = format_docs(docs)
    prompt = (
        "Используй только следующий контекст. Если в нём нет сведений для ответа, "
        "прямо скажи, что по контексту ответ дать нельзя.\n\n"
        f"Контекст:\n{context}\n\n"
        f"Вопрос: {question}\n\n"
        "Краткий ответ:"
    )
    llm = get_llm()
    msg = llm.invoke([HumanMessage(content=prompt)])
    return msg.content or ""
