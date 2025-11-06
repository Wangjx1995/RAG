# rag/llm/no_llm.py
from __future__ import annotations
from typing import List, Dict, Optional
import re

_DATE_RE = re.compile(
    r"(?:(\d{4})[年/-])?(\d{1,2})[月/-](\d{1,2})日?"
)

def _ymd_to_ja(y: Optional[str], m: str, d: str) -> str:
    if y is None:
        return f"{int(m)}月{int(d)}日"
    return f"{int(y)}年{int(m)}月{int(d)}日"

def _pick_date_near_keywords(text: str, keywords=("指名","就任","首相")) -> Optional[str]:
    for kw in keywords:
        for mkw in re.finditer(kw, text):
            start = max(0, mkw.start() - 80)
            end   = min(len(text), mkw.end() + 80)
            win = text[start:end]
            md = _DATE_RE.search(win)
            if md:
                y, m, d = md.groups()
                return _ymd_to_ja(y, m, d)
    md = _DATE_RE.search(text)
    if md:
        y, m, d = md.groups()
        return _ymd_to_ja(y, m, d)
    return None

class NoLLM:
    def chat(self, question: str, contexts: List[Dict], max_ref: int = 5, mode: str = "full") -> str:
        if mode == "concise":
            for c in contexts[: max_ref]:
                dt = _pick_date_near_keywords(c.get("text", ""))
                if dt:
                    return f"{dt}です。"
            for c in contexts[: max_ref]:
                md = _DATE_RE.search(c.get("text",""))
                if md:
                    y, m, d = md.groups()
                    return f"{_ymd_to_ja(y,m,d)}です。"
            return "該当する日付を特定できませんでした。"

        answer = "【暫定回答（LLMなし）】上位コンテキストを表示します。"
        lines = [answer, ""]
        for c in contexts[:max_ref]:
            src = f"{c.get('source')}#chunk{c.get('chunk_id')}"
            snippet = c.get("text","").replace("\n"," ")[:200]
            lines.append(f"- {src}\n  {snippet}")
        return "\n".join(lines)
