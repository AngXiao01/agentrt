"""上下文管理，从便宜到贵四层：

- 工具结果落盘（loop._shrink）：单条结果超长时全文存盘，上下文里只留预览和路径。
- microcompact：把较早的、可再生的工具结果（搜索 / 读取类）清成固定占位符，只保留最近 N 条；
  不调用模型，最便宜，所以在 autocompact 之前先试。
- autocompact：估算 token 超过 阈值 = 预算 − 缓冲 时，用模型把较早的历史总结成一条摘要。
  缓冲 13,000 tokens；预算较小时按预算的 25% 封顶。
- reactive compact：服务端报上下文超长时，用更激进的参数立刻压一次再重试。

日志只追加。压缩不改写已经写下的消息，只追加一条 compact 事件（切点、结构化摘要、读过和改过的文件，
或 microcompact 清掉的 tool_call_id）。模型看到的上下文是这条日志的投影：system + 最新摘要 + 切点之后的消息，
再套上 microcompact 的占位符。system prompt 始终保持不变（不放时间戳等易变内容），
除了这次投影变化，上下文只追加，尽量复用 KV cache。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

_NON_ASCII = re.compile(r"[^\x00-\x7f]")


def estimate_tokens(messages: list[dict]) -> int:
    """粗估：ASCII 约 4 字符 1 token，中文等约 1.5 字符 1 token，每条消息另加开销。"""
    total = 0
    for m in messages:
        text = (m.get("content") or "") + (m.get("reasoning") or m.get("reasoning_content") or "") + (
            json.dumps(m.get("tool_calls"), ensure_ascii=False) if m.get("tool_calls") else "")
        non_ascii = len(_NON_ASCII.findall(text))
        total += (len(text) - non_ascii) / 4 + non_ascii / 1.5 + 6
    return int(total)


CLEARED = "[旧的工具结果已清除；需要的话请重新调用工具]"
COMPACTABLE = {"read_file", "grep", "glob", "list_dir", "find_files", "file_stats", "find_duplicates", "bash"}  # 可再生的结果
AUTOCOMPACT_BUFFER_TOKENS = 13_000


def clear_old_results(messages: list[dict], keep_recent: int = 5) -> tuple[list[dict], int]:
    """microcompact：除最近 keep_recent 条外，其余可压缩工具的结果换成占位符。返回 (新消息, 清除条数)。
    至少保留 1 条：清光会让模型没有任何工作上下文。"""
    keep_recent = max(1, keep_recent)
    ids = [m["tool_call_id"] for m in messages if m["role"] == "tool" and m.get("name") in COMPACTABLE]
    clear = set(ids[:-keep_recent]) if len(ids) > keep_recent else set()
    out, n = [], 0
    for m in messages:
        if m["role"] == "tool" and m.get("tool_call_id") in clear and m["content"] != CLEARED:
            out.append({**m, "content": CLEARED})
            n += 1
        else:
            out.append(m)
    return out, n


def truncate_middle(text: str, max_chars: int, hint: str = "") -> str:
    if len(text) <= max_chars:
        return text
    head = text[: max_chars * 2 // 3]
    tail = text[-max_chars // 3:]
    cut = len(text) - len(head) - len(tail)
    hint = hint or "如需完整内容请用 read_file 的 offset/limit 分页读取"
    return f"{head}\n[... 已截断 {cut} 字符；{hint} ...]\n{tail}"


SUMMARY_PROMPT = (
    "请把下面这段 agent 执行历史压缩成一份工作摘要，供后续继续执行任务使用。\n"
    "按这些小节输出，没有内容的写“无”：\n"
    "## Goal\n## Constraints\n## Progress\n## Decisions\n## Next\n## Critical\n"
    "必须保留：用户的原始任务、已经完成的步骤和结论、修改过的文件、遇到的错误及原因、尚未完成的事项。\n"
    "不要编造，不要输出工具调用标签。\n\n"
)

# 摘要里的文件清单不靠模型回忆：从被压掉的工具调用里提取，并跨次压缩累加。
_READ_TOOLS = {"read_file", "grep", "glob", "list_dir", "find_files", "file_stats", "find_duplicates"}
_WRITE_TOOLS = {"write_file", "edit_file", "move_file", "copy_file", "trash"}


def _uniq(paths: list[str]) -> list[str]:
    out: list[str] = []
    for p in paths:
        if p and p not in out:
            out.append(p)
    return out


def files_touched(messages: list[dict]) -> tuple[list[str], list[str]]:
    """被摘要掉的跨度里，读过和改过的路径。bash 改了哪些文件无法从参数看出来，不猜。"""
    read: list[str] = []
    modified: list[str] = []
    for m in messages:
        for c in m.get("tool_calls") or []:
            args = c.get("arguments") or {}
            if not isinstance(args, dict):
                continue
            name = c.get("name") or ""
            if name in _READ_TOOLS:
                read.append(str(args.get("path") or args.get("pattern") or ""))
            elif name in _WRITE_TOOLS:
                for key in ("path", "src", "dst"):
                    if args.get(key):
                        modified.append(str(args[key]))
    return _uniq(read), _uniq(modified)


def render_summary(reason: str, summary: str, read_files: list[str], modified_files: list[str]) -> str:
    """投影里代替被压掉历史的那条 user 消息。文件清单是 harness 算的，不进模型摘要正文。"""
    parts = [f"[上下文已压缩（{reason}），以下是此前工作的摘要]", (summary or "").strip()]
    if read_files or modified_files:
        lines = ["## Files"]
        if read_files:
            lines.append("Read: " + ", ".join(read_files))
        if modified_files:
            lines.append("Modified: " + ", ".join(modified_files))
        parts.append("\n".join(lines))
    return "\n\n".join(p for p in parts if p)


@dataclass
class CompactPlan:
    """一次压缩的结果。view 是当时的投影，供还没写日志的调用方用；真正落盘的是其余字段。"""

    reason: str
    view: list[dict]
    first_kept_id: Optional[int] = None
    retain_none: bool = False
    summary: str = ""
    read_files: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)
    cleared_ids: list[str] = field(default_factory=list)


class ContextManager:
    # 计数方式：不跑 tokenizer，而是 “上一次 API 返回的 usage + 其后新增消息的粗估（字符数 / 4）”。
    # 服务端数的才是准的，也不依赖第三方库。
    def __init__(self, llm, max_context_tokens: int = 24000, compact_ratio: Optional[float] = None,
                 keep_recent: int = 6, tool_result_max_chars: int = 6000, enabled: bool = True,
                 mc_keep_recent: int = 5):
        self.llm = llm
        self.max_context_tokens = max_context_tokens
        # 阈值 = 预算 − 缓冲（13,000）；预算小时缓冲按 25% 封顶。compact_ratio 仅用于测试 / 对照实验覆盖
        self.compact_ratio = compact_ratio
        self.mc_keep_recent = mc_keep_recent
        self.keep_recent = keep_recent
        self.tool_result_max_chars = tool_result_max_chars
        self.enabled = enabled
        self.compactions = 0
        self.last_summary_usage: Optional[dict] = None   # 主循环取走后清空，用来给摘要调用记账
        self.last_plan: Optional[CompactPlan] = None
        self.read_files: list[str] = []       # 跨次摘要累加；resume 时由主循环从最后一条 compact 事件种回来
        self.modified_files: list[str] = []
        self._anchor: Optional[tuple[int, int]] = None   # (到这条为止的消息数, 此时服务端报告的上下文 tokens)
        self._request_anchor: Optional[tuple[int, int]] = None   # 同一次调用、不含回复的锚点（见 drop_reply）

    def note_usage(self, n_messages_sent: int, prompt_tokens: int, completion_tokens: int) -> None:
        """记下一次模型调用的真实用量：发出 n 条消息，返回后上下文 = prompt + completion，对应前 n+1 条消息。"""
        if prompt_tokens:
            self._anchor = (n_messages_sent + 1, prompt_tokens + completion_tokens)
            self._request_anchor = (n_messages_sent, prompt_tokens)

    def drop_reply(self) -> None:
        """主循环没把刚生成的回复放进上下文（被截断、只有思考）：锚点退回到只算请求本身，
        否则那几万 token 的思考会一直算在上下文里，提前触发压缩。"""
        if self._anchor:
            self._anchor = self._request_anchor

    def estimate(self, messages: list[dict]) -> int:
        """按真正发出去的消息粗估：内部消息里存着每一轮的 reasoning，但按 reasoning_replay 大多不回传，直接数会高估好几倍。"""
        wire = getattr(self.llm, "wire", None)
        return estimate_tokens(wire(messages) if wire else messages)

    def count(self, messages: list[dict]) -> int:
        if self._anchor and len(messages) >= self._anchor[0]:
            return self._anchor[1] + estimate_tokens(messages[self._anchor[0]:])
        return self.estimate(messages)

    def _safe_boundary(self, messages: list[dict], keep: int) -> int:
        """找切分点：保留区不能以 tool 结果开头，否则会和它的 assistant 调用分离。"""
        idx = max(1, len(messages) - keep)
        while idx > 1 and messages[idx]["role"] == "tool":
            idx -= 1
        return idx

    def _summarize(self, old: list[dict]) -> str:
        transcript = []
        for m in old:
            role = m["role"]
            content = m.get("content") or ""
            if role == "tool":
                content = truncate_middle(content, 1500)
                transcript.append(f"[工具结果 {m.get('name', '')}] {content}")
            else:
                calls = "".join(f"\n  调用 {c.get('name', '')}({truncate_middle(json.dumps(c.get('arguments'), ensure_ascii=False), 800)})"
                                for c in m.get("tool_calls") or [])   # native 协议下调用不在 content 里
                transcript.append(f"[{role}] {truncate_middle(content, 3000)}{calls}")
        req = [{"role": "system", "content": "你是一个负责压缩上下文的助手。"},
               {"role": "user", "content": SUMMARY_PROMPT + "\n".join(transcript)}]
        res = self.llm.chat(req, None, call_prefix="summary")
        self.last_summary_usage = res.usage or {}
        # 摘要里如果带了工具调用标签，去掉，避免被当成调用执行
        return re.sub(r"</?tool_call>", "", res.content).strip()

    def compact(self, messages: list[dict], keep: int, reason: str,
                sources: Optional[list[dict]] = None) -> Optional[list[dict]]:
        """压出投影。sources 与 messages 逐条对应，给摘要用：microcompact 的占位符里没有原文。
        保留的尾巴仍取 messages，占位符留在模型视图里。"""
        self.last_plan = None
        src = sources or messages
        if len(messages) <= keep + 2 or len(src) != len(messages):
            return None
        cut = self._safe_boundary(messages, keep)
        old = src[1:cut]
        recent = messages[cut:]
        if not old:
            return None
        summary = self._summarize(old)
        self.compactions += 1
        self._anchor = None
        span_read, span_mod = files_touched(old)
        self.read_files = _uniq(self.read_files + span_read)
        self.modified_files = _uniq(self.modified_files + span_mod)
        note = {"role": "user", "kind": "summary",
                "content": render_summary(reason, summary, self.read_files, self.modified_files)}
        kept = messages[cut] if cut < len(messages) else None
        view = [messages[0], note] + recent
        self.last_plan = CompactPlan(
            reason=reason, view=view, summary=summary,
            first_kept_id=kept.get("id") if kept else None, retain_none=kept is None,
            read_files=list(self.read_files), modified_files=list(self.modified_files))
        return view

    @property
    def threshold(self) -> int:
        if self.compact_ratio is not None:
            return int(self.max_context_tokens * self.compact_ratio)
        return self.max_context_tokens - min(AUTOCOMPACT_BUFFER_TOKENS, self.max_context_tokens // 4)

    def maybe_autocompact(self, messages: list[dict],
                          sources: Optional[list[dict]] = None) -> Optional[tuple[list[dict], str]]:
        """返回 (投影, 原因) 或 None。先试 microcompact（不调模型）；还超阈值才让模型做摘要。
        sources 是同一条视图、但工具结果是原文，只给摘要看。"""
        self.last_plan = None
        if not self.enabled or self.count(messages) < self.threshold:
            return None
        cleared, n = clear_old_results(messages, self.mc_keep_recent)
        if n:
            self._anchor = None   # 投影变了，旧的用量锚点作废，回到粗估
        if n and self.estimate(cleared) < self.threshold:
            self.last_plan = self._micro_plan(messages, cleared)
            return cleared, "microcompact"
        # 摘要看原文：microcompact 的占位符里没有工具输出，拿它做摘要等于什么都没看到
        summarized = self.compact(messages, self.keep_recent, "autocompact", sources)
        if summarized is not None:
            return summarized, "autocompact"
        if n:
            self.last_plan = self._micro_plan(messages, cleared)
            return cleared, "microcompact"
        return None

    @staticmethod
    def _micro_plan(before: list[dict], cleared: list[dict]) -> CompactPlan:
        prev = {m.get("tool_call_id"): m.get("content") for m in before if m.get("role") == "tool"}
        ids = [m["tool_call_id"] for m in cleared
               if m.get("role") == "tool" and m.get("content") == CLEARED
               and prev.get(m.get("tool_call_id")) != CLEARED]
        return CompactPlan(reason="microcompact", view=cleared, cleared_ids=ids)

    def reactive_compact(self, messages: list[dict],
                         sources: Optional[list[dict]] = None) -> Optional[list[dict]]:
        return self.compact(messages, max(2, self.keep_recent // 2), "reactive", sources)
