# rag/llm/openai_chat.py
from __future__ import annotations
import os
from typing import List, Dict
from openai import OpenAI

def _maybe_temperature_from_env(env_name: str = "OPENAI_TEMPERATURE"):
    val = os.getenv(env_name)
    if not val:
        return None
    try:
        t = float(val)
        return None if t == 1 else t
    except Exception:
        return None

class OpenAIChat:
    def __init__(self, model: str | None = None):
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL") or None
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY 未配置")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model or os.getenv("OPENAI_CHAT_MODEL") or "gpt-5-mini"
        self.temperature = _maybe_temperature_from_env("OPENAI_TEMPERATURE")

    def chat(self, question: str, contexts: List[Dict], max_ref: int = 4) -> str:
        sys = (
            "あなたは有能なアシスタントです。参考コンテキストを根拠として、"
            "事実ベースで日本語の自然な文章で簡潔に回答してください。"
            "答えが不確かなら、その旨を率直に述べてください。"
        )
        refs = "\n\n".join(
            [f"[{i+1}] {c.get('text','')}" for i, c in enumerate(contexts[:max_ref])]
        )
        user = f"質問: {question}\n\n参考コンテキスト:\n{refs}\n\n出典URLやファイル名は出力しないでください。"

        kwargs = dict(model=self.model, messages=[{"role":"system","content":sys},
                                                  {"role":"user","content":user}])
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        r = self.client.chat.completions.create(**kwargs)
        return r.choices[0].message.content.strip()
