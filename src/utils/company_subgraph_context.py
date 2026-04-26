from __future__ import annotations

from typing import Any


def _safe_label(label: str) -> str:
    return label.replace("`", "``")


def _choose_company_label(schema: dict[str, Any]) -> str | None:
    labels = [
        label
        for label, entry in schema.items()
        if isinstance(entry, dict) and entry.get("type") == "node"
    ]
    for label in labels:
        if "company" in label.lower():
            return label
    return labels[0] if labels else None


def _pick_anchor_candidates(db_manager, company_label: str, limit: int) -> list[dict[str, Any]]:
    query = f"""
    MATCH (c:`{_safe_label(company_label)}`)
    OPTIONAL MATCH (c)-[r]-()
    WITH
      c,
      count(r) AS degree,
      count(DISTINCT type(r)) AS rel_type_variety
    ORDER BY degree DESC, rel_type_variety DESC
    RETURN
      elementId(c) AS anchor_id,
      properties(c) AS anchor_props,
      degree,
      rel_type_variety
    LIMIT $limit
    """
    return db_manager.run_query(query, {"limit": limit})


def _get_subgraph_snapshot(
    db_manager,
    anchor_id: str,
    hop1_limit: int = 64,
    hop2_limit: int = 96,
) -> dict[str, Any]:
    # Берем локальный подграф вокруг anchor в 1 и 2 hops, но ограничиваем
    # объем, чтобы контекст оставался пригодным для промпта.
    query = """
    MATCH (c)
    WHERE elementId(c) = $anchor_id
    OPTIONAL MATCH (c)-[r1]-(n1)
    WITH c, collect(DISTINCT {
      rel_type: type(r1),
      node_labels: labels(n1),
      node_props: properties(n1)
    })[..$hop1_limit] AS hop1
    OPTIONAL MATCH (c)-[r_a]-(mid)-[r_b]-(n2)
    WITH c, hop1, collect(DISTINCT {
      rel_type_1: type(r_a),
      mid_labels: labels(mid),
      mid_props: properties(mid),
      rel_type_2: type(r_b),
      node2_labels: labels(n2),
      node2_props: properties(n2)
    })[..$hop2_limit] AS hop2
    RETURN properties(c) AS anchor_props, labels(c) AS anchor_labels, hop1, hop2
    """
    rows = db_manager.run_query(
        query,
        {"anchor_id": anchor_id, "hop1_limit": hop1_limit, "hop2_limit": hop2_limit},
    )
    if not rows:
        return {}
    return _sanitize_snapshot(rows[0])


def _anchor_search_needles(anchor_props: dict[str, Any]) -> list[str]:
    """Подстроки для проверки, что текст новости/статьи относится к якорной компании."""
    if not isinstance(anchor_props, dict):
        return []
    out: list[str] = []

    def _push_name_variant(raw: str) -> None:
        n = raw.strip()
        if len(n) < 4:
            return
        low = n.lower()
        out.append(low)
        first = n.split(",")[0].strip()
        if len(first) >= 4 and first.lower() != low:
            out.append(first.lower())
        t = low
        for suf in (
            " incorporated",
            " corporation",
            " corp.",
            " corp",
            " plc",
            " ltd.",
            " ltd",
            ", inc.",
            " inc.",
            " inc",
            ", inc",
        ):
            if t.endswith(suf):
                t = t[: -len(suf)].strip()
                break
        if len(t) >= 4 and t not in out:
            out.append(t)

    name = anchor_props.get("name")
    if isinstance(name, str):
        _push_name_variant(name)
    ticker = anchor_props.get("ticker")
    if isinstance(ticker, str):
        t = ticker.strip().lstrip("$").lower()
        if len(t) >= 2:
            out.append(t)
            out.append(f"${t}")
    seen: set[str] = set()
    uniq: list[str] = []
    for x in out:
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _text_blob_from_props(props: dict[str, Any]) -> str:
    if not isinstance(props, dict):
        return ""
    parts: list[str] = []
    for v in props.values():
        if isinstance(v, str):
            parts.append(v.lower())
        elif isinstance(v, (int, float)):
            parts.append(str(v).lower())
    return " ".join(parts)


def _props_match_company(needles: list[str], props: dict[str, Any]) -> bool:
    if not needles:
        return True
    blob = _text_blob_from_props(props)
    if not blob.strip():
        return False
    return any(n in blob for n in needles)


def _is_news_like_node(labels: Any, rel_type: str | None) -> bool:
    lab = " ".join(labels or []).lower() if labels else ""
    rt = (rel_type or "").lower()
    return "news" in lab or "article" in lab or "press" in lab or "news" in rt


def _filter_hop1_relevant(anchor_props: dict[str, Any], hop1: list[Any]) -> list[dict[str, Any]]:
    needles = _anchor_search_needles(anchor_props)
    kept: list[dict[str, Any]] = []
    for item in hop1:
        if not isinstance(item, dict) or not item.get("rel_type"):
            continue
        labels = item.get("node_labels")
        rel_type = item.get("rel_type")
        props = item.get("node_props") or {}
        if _is_news_like_node(labels, str(rel_type) if rel_type else None):
            if not needles or not _props_match_company(needles, props):
                continue
        kept.append(item)

    def _sort_key(row: dict[str, Any]) -> tuple[str, str]:
        p = row.get("node_props") or {}
        h = p.get("headline") or p.get("title") or ""
        return (str(row.get("rel_type") or ""), str(h))

    kept.sort(key=_sort_key)
    return kept


def _filter_hop2_relevant(anchor_props: dict[str, Any], hop2: list[Any]) -> list[dict[str, Any]]:
    needles = _anchor_search_needles(anchor_props)
    kept: list[dict[str, Any]] = []
    for item in hop2:
        if not isinstance(item, dict):
            continue
        r1 = item.get("rel_type_1")
        r2 = item.get("rel_type_2")
        mid_labels = item.get("mid_labels")
        mid_props = item.get("mid_props") or {}
        n2_labels = item.get("node2_labels")
        n2_props = item.get("node2_props") or {}
        if _is_news_like_node(mid_labels, str(r1) if r1 else None):
            if not needles or not _props_match_company(needles, mid_props):
                continue
        if _is_news_like_node(n2_labels, str(r2) if r2 else None):
            if not needles or not _props_match_company(needles, n2_props):
                continue
        if not item.get("rel_type_1") or not item.get("rel_type_2"):
            continue
        kept.append(item)

    def _sort_key2(row: dict[str, Any]) -> tuple[str, str, str]:
        p2 = row.get("node2_props") or {}
        h = p2.get("headline") or p2.get("title") or ""
        return (str(row.get("rel_type_1") or ""), str(row.get("rel_type_2") or ""), str(h))

    kept.sort(key=_sort_key2)
    return kept


def _prune_snapshot_to_anchor_relevant(snapshot: dict[str, Any]) -> dict[str, Any]:
    """
    Убирает из снапшота новости/статьи, в тексте которых нет имени или тикера якоря
    (типичный шум при широком матчинге NEWS_ABOUT_COMPANY). Остальные соседи сохраняются.
    """
    anchor_props = snapshot.get("anchor_props") or {}
    hop1_raw = snapshot.get("hop1") or []
    hop2_raw = snapshot.get("hop2") or []
    hop1 = _filter_hop1_relevant(anchor_props, hop1_raw)[:20]
    hop2 = _filter_hop2_relevant(anchor_props, hop2_raw)[:30]
    out = dict(snapshot)
    out["hop1"] = hop1
    out["hop2"] = hop2
    return out


def _should_drop_key(key: str) -> bool:
    lowered = key.lower()
    if "embedding" in lowered:
        return True
    return lowered in {"vector", "vectors", "embedding", "embeddings"}


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for k, v in value.items():
            if _should_drop_key(str(k)):
                continue
            cleaned[k] = _sanitize_value(v)
        return cleaned
    if isinstance(value, list):
        return [_sanitize_value(v) for v in value]
    return value


def _sanitize_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    return _sanitize_value(snapshot)


def _props_preview(props: dict[str, Any], max_items: int = 4) -> str:
    if not isinstance(props, dict) or not props:
        return "{}"
    pairs = []
    for idx, (k, v) in enumerate(props.items()):
        if idx >= max_items:
            break
        pairs.append(f"{k}={v!r}")
    return "{ " + ", ".join(pairs) + " }"


def _snapshot_to_text(snapshot: dict[str, Any]) -> str:
    anchor_labels = snapshot.get("anchor_labels") or []
    anchor_props = snapshot.get("anchor_props") or {}
    hop1 = snapshot.get("hop1") or []
    hop2 = snapshot.get("hop2") or []

    lines = [
        "ANCHOR:",
        f"- labels: {anchor_labels}",
        f"- props: {_props_preview(anchor_props)}",
        "",
        "HOP1 RELATIONS:",
    ]

    for item in hop1[:20]:
        if not item or not item.get("rel_type"):
            continue
        lines.append(
            f"- ({anchor_labels}) -[:{item.get('rel_type')}]- "
            f"({item.get('node_labels')}) {_props_preview(item.get('node_props') or {})}"
        )

    lines.append("")
    lines.append("HOP2 RELATIONS:")
    for item in hop2[:30]:
        if not item:
            continue
        if not item.get("rel_type_1") or not item.get("rel_type_2"):
            continue
        lines.append(
            f"- ({anchor_labels}) -[:{item.get('rel_type_1')}]- "
            f"({item.get('mid_labels')}) -[:{item.get('rel_type_2')}]- "
            f"({item.get('node2_labels')})"
        )

    return "\n".join(lines).strip()


def _build_debug_subgraph_cypher() -> str:
    return """
MATCH (c)
WHERE elementId(c) = $anchor_id
OPTIONAL MATCH (c)-[r1]-(n1)
WITH c, collect(DISTINCT {
  rel_type: type(r1),
  node_labels: labels(n1),
  node_props: properties(n1)
})[..20] AS hop1
OPTIONAL MATCH (c)-[r_a]-(mid)-[r_b]-(n2)
WITH c, hop1, collect(DISTINCT {
  rel_type_1: type(r_a),
  mid_labels: labels(mid),
  mid_props: properties(mid),
  rel_type_2: type(r_b),
  node2_labels: labels(n2),
  node2_props: properties(n2)
})[..30] AS hop2
RETURN properties(c) AS anchor_props, labels(c) AS anchor_labels, hop1, hop2
""".strip()


def _extract_useful_context(snapshot: dict[str, Any], max_lines: int = 16) -> str:
    """
    Оставляем только сигналы, полезные для ответа:
    сущности, события, метрики, тональность/риск, временные и гео-признаки.
    """
    anchor_props = snapshot.get("anchor_props") or {}
    hop1 = snapshot.get("hop1") or []
    hop2 = snapshot.get("hop2") or []

    focus_keys = {
        "name",
        "title",
        "headline",
        "description",
        "summary",
        "text",
        "content",
        "body",
        "article_text",
        "snippet",
        "ticker",
        "sector",
        "industry",
        "country",
        "region",
        "date",
        "year",
        "amount",
        "value",
        "revenue",
        "profit",
        "risk",
        "sentiment",
        "score",
        "impact",
        "category",
        "status",
    }
    banned_keys = {"embedding", "embeddings", "vector", "vectors"}

    def pick_props(props: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(props, dict):
            return {}
        out: dict[str, Any] = {}
        for k, v in props.items():
            lk = str(k).lower()
            if lk in banned_keys or "embedding" in lk:
                continue
            key_is_focus = lk in focus_keys or any(
                token in lk for token in ("headline", "title", "description", "summary", "text", "content")
            )
            if key_is_focus and v not in (None, ""):
                out[k] = v
        if not out:
            for k, v in props.items():
                lk = str(k).lower()
                if lk in banned_keys or "embedding" in lk:
                    continue
                if isinstance(v, (str, int, float)) and v not in (None, ""):
                    out[k] = v
                if len(out) >= 3:
                    break
        return out

    lines = ["KEY CONTEXT SIGNALS:"]
    anchor_focus = pick_props(anchor_props)
    if anchor_focus:
        lines.append(f"- anchor: {anchor_focus}")

    seen_signal_keys: set[str] = set()
    for item in hop1:
        if len(lines) >= max_lines:
            break
        if not item or not item.get("rel_type"):
            continue
        rel_type = item.get("rel_type")
        props = pick_props(item.get("node_props") or {})
        if props:
            signal_key = f"{rel_type}|{props}"
            if signal_key in seen_signal_keys:
                continue
            seen_signal_keys.add(signal_key)
            lines.append(f"- signal: {rel_type} -> {props}")

    for item in hop2:
        if len(lines) >= max_lines:
            break
        if not item:
            continue
        rel1 = item.get("rel_type_1")
        rel2 = item.get("rel_type_2")
        node2_props = pick_props(item.get("node2_props") or {})
        if rel1 and rel2 and node2_props:
            signal_key = f"{rel1}->{rel2}|{node2_props}"
            if signal_key in seen_signal_keys:
                continue
            seen_signal_keys.add(signal_key)
            lines.append(f"- chain: {rel1} -> {rel2} -> {node2_props}")

    return "\n".join(lines).strip()


def build_company_subgraph_contexts(
    db_manager,
    schema: dict[str, Any],
    anchors_limit: int = 3,
) -> list[dict[str, Any]]:
    """
    Возвращает набор контекстов подграфа компаний для deep-analytics вопросов:
      [
        {
          "anchor_id": "...",
          "anchor_props": {...},
          "subgraph_context": "text block"
        }
      ]
    """
    company_label = _choose_company_label(schema)
    if not company_label:
        return []

    anchors = _pick_anchor_candidates(db_manager, company_label, anchors_limit)
    out: list[dict[str, Any]] = []
    for anchor in anchors:
        anchor_id = anchor.get("anchor_id")
        if not anchor_id:
            continue
        snapshot = _get_subgraph_snapshot(db_manager, anchor_id)
        if not snapshot:
            continue
        snapshot = _prune_snapshot_to_anchor_relevant(snapshot)
        context_text = _snapshot_to_text(snapshot)
        if not context_text:
            continue
        out.append(
            {
                "company_label": company_label,
                "anchor_id": anchor_id,
                "anchor_props": snapshot.get("anchor_props") or {},
                "subgraph_context": context_text,
                "useful_context": _extract_useful_context(snapshot),
                "debug_cypher": _build_debug_subgraph_cypher(),
                "debug_params": {"anchor_id": anchor_id},
            }
        )
    return out

