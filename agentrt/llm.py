"""LLM 客户端：OpenAI 兼容接口（vLLM）+ 用于测试的脚本化模型。

两种工具调用协议：
- text（默认）：工具说明写进 system prompt，模型用 <tool_call>{...}</tool_call> 输出调用。
  不依赖 vLLM 的 tool-call-parser，任何聊天模型都能用。
- native：走 OpenAI 的 tools / tool_calls 字段，需要 vLLM 启动时开
  --enable-auto-tool-choice --tool-call-parser <parser>。
"""
from __future__ import annotations

import json
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional


class LLMError(Exception):
    """不可重试的模型错误。"""


class RetryableError(LLMError):
    """网络抖动、5xx、429 等可重试错误。"""


class ContextLengthError(LLMError):
    """上下文超长：由上层做 reactive compact 后重试。"""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ChatResult:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    finish_reason: str = ""


# ---------------------------------------------------------------- 文本协议解析

_TOOL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.S)
# Gemma 原生调用格式 call:name{...}（外层特殊 token 被 vLLM 去掉了），不当成最终答复
_NATIVE_CALL_RE = re.compile(r"(?:^|\n)\s*call:([A-Za-z_]\w*)\s*\{")


def parse_text_tool_calls(text: str, id_prefix: str) -> tuple[list[ToolCall], list[str]]:
    """从模型文本里解析 <tool_call> 块。解析失败的块不丢弃，返回错误信息给模型看。"""
    calls, errors = [], []
    for i, raw in enumerate(_TOOL_RE.findall(text or "")):
        body = _FENCE_RE.sub("", raw.strip())
        try:
            obj = json.loads(body)
            name = obj["name"]
            args = obj.get("arguments", {}) or {}
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(args, dict):
                raise ValueError("arguments 必须是 JSON 对象")
            calls.append(ToolCall(id=f"{id_prefix}_{i}", name=str(name), arguments=args))
        except Exception as e:  # noqa: BLE001
            errors.append(f"第 {i + 1} 个 <tool_call> 无法解析（{e}）：{body[:200]}")
    if "<tool_call>" in (text or "") and "</tool_call>" not in text:
        errors.append("检测到 <tool_call> 但缺少 </tool_call> 闭合标签")
    if not calls and not errors and (m := _NATIVE_CALL_RE.search(text or "")):
        errors.append(f"检测到 call:{m.group(1)}{{...}} 形式的调用，这里不支持；请改用 "
                      f'<tool_call>{{"name": "{m.group(1)}", "arguments": {{...}}}}</tool_call>')
    return calls, errors


# ---------------------------------------------------------------- 发给服务端前的消息整形

def to_wire(messages: list[dict], native_tools: bool, merge_system: bool) -> list[dict]:
    """内部消息 -> 接口消息。

    - text 协议下，role=tool 的结果转成 user 消息（带 <tool_result> 标签）。
    - 连续同角色消息合并：Gemma 一类模板要求 user/assistant 严格交替。
    - merge_system=True 时把 system 并进第一条 user（部分模板不支持 system 角色）。
    """
    out: list[dict] = []
    for m in messages:
        role = m["role"]
        if role == "tool" and not native_tools:
            content = (f'<tool_result id="{m["tool_call_id"]}" name="{m.get("name", "")}">\n'
                       f'{m["content"]}\n</tool_result>')
            wm = {"role": "user", "content": content}
        elif role == "assistant" and native_tools and m.get("tool_calls"):
            wm = {"role": "assistant", "content": m.get("content") or "",
                  "tool_calls": [{"id": c["id"], "type": "function",
                                  "function": {"name": c["name"],
                                               "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                                 for c in m["tool_calls"]]}
        elif role == "tool":
            wm = {"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]}
        else:
            wm = {"role": role, "content": m.get("content") or ""}
        if (out and wm["role"] == out[-1]["role"] and wm["role"] in ("user", "assistant")
                and "tool_calls" not in wm and "tool_calls" not in out[-1]):
            out[-1] = {**out[-1], "content": out[-1]["content"] + "\n\n" + wm["content"]}
        else:
            out.append(wm)
    if merge_system and out and out[0]["role"] == "system":
        sys_msg = out.pop(0)["content"]
        if out and out[0]["role"] == "user":
            out[0] = {**out[0], "content": sys_msg + "\n\n" + out[0]["content"]}
        else:
            out.insert(0, {"role": "user", "content": sys_msg})
    return out


# ---------------------------------------------------------------- vLLM / OpenAI 兼容客户端

class OpenAICompatLLM:
    def __init__(self, base_url: str, model: Optional[str] = None, api_key: str = "EMPTY",
                 timeout: float = 180, max_retries: int = 4, native_tools: bool = False,
                 merge_system: bool = False, temperature: float = 0.2, max_tokens: int = 2048,
                 sleep: Callable[[float], None] = time.sleep):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.native_tools = native_tools
        self.merge_system = merge_system
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._sleep = sleep
        self.max_model_len: Optional[int] = None
        self.model = self._resolve_model(model)

    def _request(self, method: str, path: str, payload: Optional[dict] = None) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="ignore")
            if e.code == 400 and re.search(r"maximum context length|context length|too long", body, re.I):
                raise ContextLengthError(body[:500]) from e
            if e.code in (408, 429) or e.code >= 500:
                raise RetryableError(f"HTTP {e.code}: {body[:300]}") from e
            raise LLMError(f"HTTP {e.code}: {body[:500]}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise RetryableError(str(e)) from e

    def _with_retry(self, fn):
        """指数退避 + 抖动；只重试 RetryableError。"""
        for attempt in range(self.max_retries + 1):
            try:
                return fn()
            except RetryableError:
                if attempt == self.max_retries:
                    raise
                delay = min(30.0, 1.0 * 2 ** attempt) * (0.5 + random.random())
                self._sleep(delay)

    def _resolve_model(self, wanted: Optional[str]) -> str:
        """取 /v1/models：确认连通和鉴权，顺便拿到 vLLM 报告的 max_model_len。"""
        data = self._with_retry(lambda: self._request("GET", "/models"))
        entries = data.get("data", []) or []
        if not entries:
            raise LLMError("服务端 /v1/models 没有返回模型")
        entry = next((m for m in entries if m.get("id") == wanted), None) if wanted else entries[0]
        if entry is None:
            raise LLMError(f"服务端没有模型 {wanted}，可用：{[m.get('id') for m in entries]}")
        if isinstance(entry.get("max_model_len"), int):
            self.max_model_len = entry["max_model_len"]
        return entry["id"]

    def context_budget(self, requested: int, margin: int = 512) -> int:
        """上下文预算不能超过服务端的 max_model_len 减去输出长度。"""
        if not self.max_model_len:
            return requested
        return max(1024, min(requested, self.max_model_len - self.max_tokens - margin))

    def chat(self, messages: list[dict], tools_schema: Optional[list[dict]] = None,
             call_prefix: str = "c") -> ChatResult:
        payload = {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                   "messages": to_wire(messages, self.native_tools, self.merge_system)}
        if self.native_tools and tools_schema:
            payload["tools"] = tools_schema
            payload["tool_choice"] = "auto"
        if not self.native_tools:
            payload["stop"] = ["<tool_result"]  # 防止模型自己编造工具结果
        data = self._with_retry(lambda: self._request("POST", "/chat/completions", payload))
        choice = data["choices"][0]
        msg = choice.get("message", {})
        content = msg.get("content") or ""
        usage = data.get("usage", {}) or {}
        if self.native_tools:
            calls, errors = [], []
            for i, tc in enumerate(msg.get("tool_calls") or []):
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                    calls.append(ToolCall(id=tc.get("id") or f"{call_prefix}_{i}", name=fn.get("name", ""),
                                          arguments=args))
                except json.JSONDecodeError as e:
                    errors.append(f"工具 {fn.get('name')} 的 arguments 不是合法 JSON：{e}")
            if not calls:  # 有些 parser 失败时会把调用留在文本里
                calls, errors2 = parse_text_tool_calls(content, call_prefix)
                errors += errors2
        else:
            calls, errors = parse_text_tool_calls(content, call_prefix)
        return ChatResult(content=content, tool_calls=calls, parse_errors=errors, usage=usage,
                          finish_reason=choice.get("finish_reason") or "")


# ---------------------------------------------------------------- 测试用脚本化模型

class ScriptedLLM:
    """按顺序返回预设回复。每一步可以是字符串、异常实例，或 callable(messages)->str。"""

    def __init__(self, steps: list, native_tools: bool = False):
        self.steps = list(steps)
        self.calls: list[list[dict]] = []
        self.native_tools = native_tools
        self.model = "scripted"

    def chat(self, messages, tools_schema=None, call_prefix: str = "c") -> ChatResult:
        self.calls.append([dict(m) for m in messages])
        if not self.steps:
            return ChatResult(content="（脚本已结束）")
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        text = step(messages) if callable(step) else step
        calls, errors = parse_text_tool_calls(text, call_prefix)
        return ChatResult(content=text, tool_calls=calls, parse_errors=errors,
                          usage={"prompt_tokens": sum(len(m.get("content") or "") for m in messages) // 4,
                                 "completion_tokens": len(text) // 4})
