"""OpenRouter 后端：key 与模型沿用 ayaka_game 的配置约定，key 本身不会被打印或写入日志。

key 查找顺序：环境变量 OPENROUTER_API_KEY → OPENROUTER_API_KEY_FILE（默认 ~/.openrouter_key，
和 ayaka_game 共用同一个文件）。模型：--model → AGENTRT_MODEL → OPENROUTER_MODEL（环境变量，
或 ayaka_game/.env 里的值）→ 默认 qwen/qwen3.7-flash。
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from .llm import LLMError, OpenAICompatLLM

BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "qwen/qwen3.7-flash"
_KEY_RE = re.compile(r"sk-or-v\d+-[A-Za-z0-9]+")
AYAKA_ENV = Path(os.environ.get("AYAKA_ENV", Path.home() / "Desktop" / "ayaka_game" / ".env"))


def _dotenv(path: Path) -> dict:
    out = {}
    try:
        for line in path.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip("'\"")
    except OSError:
        pass
    return out


def read_key() -> str:
    raw = os.environ.get("OPENROUTER_API_KEY", "")
    if not raw:
        env = _dotenv(AYAKA_ENV)
        f = Path(os.environ.get("OPENROUTER_API_KEY_FILE") or env.get("OPENROUTER_API_KEY_FILE")
                 or "~/.openrouter_key").expanduser()
        try:
            raw = f.read_text()
        except OSError:
            raise LLMError(f"找不到 OpenRouter key：设置 OPENROUTER_API_KEY，或把 key 放进 {f}") from None
    m = _KEY_RE.search(raw)
    key = m.group(0) if m else raw.strip().strip("'\"").removeprefix("Bearer ").strip()
    if not key:
        raise LLMError("OpenRouter key 为空")
    return key


def pick_model(explicit: Optional[str] = None) -> str:
    for cand in (explicit, os.environ.get("AGENTRT_MODEL"), os.environ.get("OPENROUTER_MODEL"),
                 _dotenv(AYAKA_ENV).get("OPENROUTER_MODEL")):
        if cand and "/" in cand:  # ayaka 的 cheapest / cheapest-free 是它自己的魔法值，这里不认
            return cand
    return DEFAULT_MODEL


class OpenRouterLLM(OpenAICompatLLM):
    backend = "openrouter"

    def __init__(self, model: Optional[str] = None, **kw):
        self._wanted = pick_model(model)
        super().__init__(os.environ.get("OPENROUTER_BASE_URL", BASE_URL), model=self._wanted,
                         api_key=read_key(), **kw)

    def _resolve_model(self, wanted):
        return self._wanted  # OpenRouter 有几百个模型，不预先拉列表；模型 id 不对会在第一次请求时报错
