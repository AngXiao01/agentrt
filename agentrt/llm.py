"""LLM 客户端：OpenAI 兼容接口（vLLM）+ 用于测试的脚本化模型。

两种工具调用协议：
- text（默认）：工具说明写进 system prompt，模型用 <tool_call>{...}</tool_call> 输出调用。
  不依赖 vLLM 的 tool-call-parser，任何聊天模型都能用。
- native：走 OpenAI 的 tools / tool_calls 字段，需要 vLLM 启动时开
  --enable-auto-tool-choice --tool-call-parser <parser>。
"""
from __future__ import annotations

import http.client
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
    reasoning: str = ""                       # reasoning_content / reasoning / <think>…</think>
    reasoning_details: Optional[list] = None  # OpenRouter 的结构化 reasoning，回传时原样带回


# ---------------------------------------------------------------- 用量归一

def normalize_usage(u: Optional[dict]) -> dict:
    """各家 usage 字段不一样，统一成 prompt / completion / cached / reasoning。
    - OpenAI / vLLM：prompt_tokens_details.cached_tokens（vLLM 要开 --enable-prompt-tokens-details）
    - DeepSeek：prompt_cache_hit_tokens / prompt_cache_miss_tokens（没有 prompt_tokens_details，只看前者会一直是 0）
    - reasoning：completion_tokens_details.reasoning_tokens（包含在 completion_tokens 里）"""
    u = u or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or u.get("prompt_cache_hit_tokens") or 0
    return {"prompt": int(u.get("prompt_tokens") or 0), "completion": int(u.get("completion_tokens") or 0),
            "cached": int(cached), "reasoning": int((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)}


_THINK_RE = re.compile(r"^\s*(?:<think>)?(.*?)</think>\s*", re.S)


def split_think(content: str) -> tuple[str, str]:
    """没开 reasoning parser 的 vLLM 会把思考留在正文里（<think>…</think>）：拆出来，不然会污染工具调用解析和最终答复。"""
    if "</think>" not in (content or ""):
        return content or "", ""
    m = _THINK_RE.match(content)
    if not m:
        return content, ""
    return content[m.end():], m.group(1).strip()


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

REASONING_REPLAY = ("none", "turn", "all")


def reasoning_keep_from(messages: list[dict], policy: str) -> int:
    """从哪条消息开始回传 reasoning。

    - none：都不回传（大多数 harness 的默认做法）。
    - turn：只回传“最后一条真正的用户消息”之后的 assistant reasoning，也就是当前这一轮工具调用链里的思考。
      这是 DeepSeek V3.2 起 thinking-in-tool-use 的规则（同一轮内必须带回，新一轮丢掉），Qwen3 的 chat template
      也是这样渲染的。harness 自己插的提示（kind=harness）和压缩摘要不算新一轮。
    - all：全部回传（Kimi K2 Thinking / MiniMax M2 这类 interleaved thinking 模型建议保留全部历史思考）。
    """
    if policy == "all":
        return 0
    if policy != "turn":
        return len(messages)
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m["role"] == "user" and m.get("kind") not in ("harness", "summary"):
            return i
    return 0


def to_wire(messages: list[dict], native_tools: bool, merge_system: bool, reasoning_replay: str = "turn") -> list[dict]:
    """内部消息 -> 接口消息。

    - text 协议下，role=tool 的结果转成 user 消息（带 <tool_result> 标签）。
    - 连续同角色消息合并：Gemma 一类模板要求 user/assistant 严格交替。
    - merge_system=True 时把 system 并进第一条 user（部分模板不支持 system 角色）。
    - reasoning 按 reasoning_replay 策略放进 assistant 消息的 reasoning_content 字段（DeepSeek / vLLM 模板认这个字段）。
    """
    out: list[dict] = []
    keep_from = reasoning_keep_from(messages, reasoning_replay)
    for idx, m in enumerate(messages):
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
        if role == "assistant" and idx >= keep_from and m.get("reasoning"):
            wm["reasoning_content"] = m["reasoning"]
            if m.get("reasoning_details"):
                wm["reasoning_details"] = m["reasoning_details"]
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
                 sleep: Callable[[float], None] = time.sleep, reasoning_replay: str = "turn",
                 extra_body: Optional[dict] = None, max_backoff: float = 30.0,
                 think_off: Optional[dict] = None, deadline: Optional[float] = None,
                 clock: Callable[[], float] = time.time):
        """think_off：单次关掉思考时并进请求体的字段（同名覆盖 extra_body），如 OpenRouter 的
        {"reasoning": {"enabled": False}}；None 表示这个端点没法单次关思考。
        deadline：任务的绝对截止时间（epoch 秒），用来给单次请求的超时封顶，见 _timeout。"""
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_backoff = max_backoff
        self.native_tools = native_tools
        self.merge_system = merge_system
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._sleep = sleep
        if reasoning_replay not in REASONING_REPLAY:
            raise ValueError(f"reasoning_replay 必须是 {REASONING_REPLAY}")
        self.reasoning_replay = reasoning_replay
        self.extra_body = dict(extra_body or {})   # 例如 {"thinking": {"type": "enabled"}}、{"chat_template_kwargs": {...}}
        self.think_off = dict(think_off) if think_off else None
        self.deadline = deadline
        self._clock = clock
        self.max_model_len: Optional[int] = None
        self.model = self._resolve_model(model)

    def _request(self, method: str, path: str, payload: Optional[dict] = None) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=self._timeout()) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="ignore")
            if e.code == 400 and re.search(r"maximum context length|context length|too long", body, re.I):
                raise ContextLengthError(body[:500]) from e
            if e.code in (408, 429) or e.code >= 500:
                raise RetryableError(f"HTTP {e.code}: {body[:300]}") from e
            raise LLMError(f"HTTP {e.code}: {body[:500]}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException, OSError) as e:
            # 读响应中途断开（IncompleteRead、SSL EOF）不一定被 urllib 包成 URLError，一并按可重试处理
            raise RetryableError(f"{type(e).__name__}: {e}") from e
        except json.JSONDecodeError as e:   # 响应体被截断
            raise RetryableError(f"响应不是合法 JSON：{e}") from e

    def _timeout(self) -> float:
        """单次请求的超时。有截止时间时最多等剩余时间的一半、至少 60 秒（对照 Hermes 的 cap_to_run_budget）：
        一次卡住的请求不会吃掉整个时限，到点前的收尾提示也还来得及发出去。"""
        if self.deadline is None:
            return self.timeout
        return min(self.timeout, max(60.0, (self.deadline - self._clock()) / 2))

    def _with_retry(self, fn):
        """指数退避 + 抖动；只重试 RetryableError。"""
        for attempt in range(self.max_retries + 1):
            try:
                return fn()
            except RetryableError:
                if attempt == self.max_retries:
                    raise
                delay = min(self.max_backoff, 1.0 * 2 ** attempt) * (0.5 + random.random())
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

    def wire(self, messages: list[dict]) -> list[dict]:
        """这次请求真正发出去的消息（主循环据此判断前缀有没有被改写，见 loop._chat）。"""
        return to_wire(messages, self.native_tools, self.merge_system, self.reasoning_replay)

    def params(self) -> dict:
        return {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                "native_tools": self.native_tools, "merge_system": self.merge_system,
                "reasoning_replay": self.reasoning_replay, "extra_body": self.extra_body,
                "think_off": self.think_off}

    def chat(self, messages: list[dict], tools_schema: Optional[list[dict]] = None,
             call_prefix: str = "c", think: bool = True) -> ChatResult:
        """think=False：这一次请求关掉思考（并进 think_off；没配 think_off 时照常请求）。"""
        payload = {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                   **self.extra_body, **({} if think else self.think_off or {}), "messages": self.wire(messages)}
        if self.native_tools and tools_schema:
            payload["tools"] = tools_schema
            payload["tool_choice"] = "auto"
        if not self.native_tools:
            payload["stop"] = ["<tool_result"]  # 防止模型自己编造工具结果
        data = self._with_retry(lambda: self._request("POST", "/chat/completions", payload))
        choice = data["choices"][0]
        msg = choice.get("message", {})
        content, think = split_think(msg.get("content") or "")
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or think
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
                          finish_reason=choice.get("finish_reason") or "", reasoning=reasoning or "",
                          reasoning_details=msg.get("reasoning_details") or None)


# ---------------------------------------------------------------- 测试用脚本化模型

class ScriptedLLM:
    """按顺序返回预设回复。每一步可以是字符串、{"content", "reasoning"}、异常实例，或 callable(messages)->上述之一。"""

    def __init__(self, steps: list, native_tools: bool = False, reasoning_replay: str = "turn",
                 think_off: Optional[dict] = None):
        self.steps = list(steps)
        self.calls: list[list[dict]] = []
        self.wires: list[list[dict]] = []
        self.thinks: list[bool] = []   # 每次调用是否开着思考
        self.native_tools = native_tools
        self.reasoning_replay = reasoning_replay
        self.think_off = think_off
        self.model = "scripted"

    def wire(self, messages):
        return to_wire(messages, self.native_tools, False, self.reasoning_replay)

    def chat(self, messages, tools_schema=None, call_prefix: str = "c", think: bool = True) -> ChatResult:
        self.calls.append([dict(m) for m in messages])
        self.wires.append(self.wire(messages))
        self.thinks.append(think)
        if not self.steps:
            return ChatResult(content="（脚本已结束）")
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        step = step(messages) if callable(step) else step
        text, reasoning = (step.get("content", ""), step.get("reasoning", "")) if isinstance(step, dict) else (step, "")
        finish = step.get("finish_reason", "") if isinstance(step, dict) else ""
        calls, errors = parse_text_tool_calls(text, call_prefix)
        return ChatResult(content=text, tool_calls=calls, parse_errors=errors, reasoning=reasoning,
                          finish_reason=finish,
                          usage={"prompt_tokens": sum(len(m.get("content") or "") for m in messages) // 4,
                                 "completion_tokens": len(text) // 4})
