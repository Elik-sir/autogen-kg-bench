"""
Патч hipporag OpenAI-эмбеддера для бенчмарка.

1) Убирает `ipdb.set_trace()` в except (иначе Docker / неинтерактивный режим ломается).
2) Заменяет `OpenAI(base_url=...)` на клиент с явным api_key и заголовками OpenRouter:
   upstream не передаёт api_key; в части окружений это даёт обрыв / Connection error.
"""
from __future__ import annotations

import re
import site
from pathlib import Path

_IPDB_BLOCK = re.compile(
    r"(?m)^(?P<ind>[ \t]*)except:\s*\r?\n(?P=ind)[ \t]*import ipdb; ipdb\.set_trace\(\)\s*\r?\n"
)

_OAI_CLIENT = re.compile(
    r"if self\.global_config\.azure_embedding_endpoint is None:\s*\n"
    r"\s*self\.client = OpenAI\(\s*\n"
    r"\s*base_url=self\.global_config\.embedding_base_url\s*\n"
    r"\s*\)",
    re.MULTILINE,
)

_OAI_CLIENT_REPL = """if self.global_config.azure_embedding_endpoint is None:
            import os as _os
            _key = (
                _os.environ.get("OPENAI_API_KEY")
                or _os.environ.get("OPENROUTER_API_KEY")
                or ""
            ).strip()
            _kw = {
                "base_url": self.global_config.embedding_base_url,
                "api_key": _key or None,
            }
            _ref = (_os.environ.get("OPENROUTER_HTTP_REFERER") or "").strip()
            _title = (_os.environ.get("OPENROUTER_APP_TITLE") or "").strip()
            _hdr = {}
            if _ref:
                _hdr["HTTP-Referer"] = _ref
            if _title:
                _hdr["X-Title"] = _title
            if _hdr:
                _kw["default_headers"] = _hdr
            self.client = OpenAI(**_kw)
"""


def main() -> int:
    for root in site.getsitepackages():
        path = Path(root) / "hipporag" / "embedding_model" / "OpenAI.py"
        if not path.is_file():
            continue
        original = path.read_text(encoding="utf-8")
        text = original
        # Старый вариант патча съедал перевод строки перед `else:`.
        text = text.replace(
            "self.client = OpenAI(**_kw)else:",
            "self.client = OpenAI(**_kw)\n        else:",
        )
        new_text, n_ipdb = _IPDB_BLOCK.subn(
            lambda m: f'{m.group("ind")}except Exception:\n{m.group("ind")}    raise\n',
            text,
        )
        if n_ipdb == 0 and "ipdb.set_trace()" in text:
            print(f"WARN: ipdb still present but pattern did not match: {path}", flush=True)
            return 1
        text = new_text
        n_oai = 0
        if _OAI_CLIENT.search(text):
            text, n_oai = _OAI_CLIENT.subn(_OAI_CLIENT_REPL, text, count=1)
        if text != original:
            path.write_text(text, encoding="utf-8")
            print(
                f"Patched {path} (ipdb_blocks={n_ipdb}, openai_client={n_oai})",
                flush=True,
            )
        return 0
    print("hipporag embedding_model/OpenAI.py not found", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
