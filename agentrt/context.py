"""上下文管理，对照 Claude Code 源码 src/services/compact/ 和 src/utils/toolResultStorage.ts：

- 工具结果落盘（loop._shrink ↔ toolResultStorage.persistToolResult）：单条结果超长时全文存盘，上下文里只留预览和路径。
- microcompact（↔ microCompact.ts）：把较早的、可再生的工具结果（搜索 / 读取类）清成固定占位符，只保留最近 N 条；
  不调用模型，最便宜，所以在 autocompact 之前先试。
- autocompact（↔ autoCompact.ts）：估算 token 超过 阈值 = 预算 − 缓冲 时，用模型把较早的历史总结成一条摘要。
  源码的缓冲是 13,000 tokens（AUTOCOMPACT_BUFFER_TOKENS）；预算较小时按预算的 25% 封顶。
- reactive compact（↔ reactiveCompact.ts）：服务端报上下文超长时，用更激进的参数立刻压一次再重试。
源码里还有 snip / context collapse，这里没有实现。

system prompt 始终保持不变（不放时间戳等易变内容），压缩只替换中间的历史，
其余时候上下文只追加不修改，尽量复用 KV cache。
"""
from __future__ import annotations

import json
import re
from typing import Optional

_NON_ASCII = re.compile(r"[^\x00-\x7f]")


def estimate_tokens(messages: list[dict]) -> int:
    """粗估：ASCII 约 4 字符 1 token，中文等约 1.5 字符 1 token，每条消息另加开销。"""
    total = 0
    for m in messages:
        text = (m.get("content") or "") + (json.dumps(m.get("tool_calls"), ensure_ascii=False)
                                           if m.get("tool_calls") else "")
        non_ascii = len(_NON_ASCII.findall(text))
        total += (len(text) - non_ascii) / 4 + non_ascii / 1.5 + 6
    return int(total)


CLEARED = "[旧的工具结果已清除；需要的话请重新调用工具]"
COMPACTABLE = {"read_file", "grep", "list_dir", "find_files", "file_stats", "find_duplicates", "bash"}  # 可再生的结果
AUTOCOMPACT_BUFFER_TOKENS = 13_000


def clear_old_results(messages: list[dict], keep_recent: int = 5) -> tuple[list[dict], int]:
    """microcompact：除最近 keep_recent 条外，其余可压缩工具的结果换成占位符。返回 (新消息, 清除条数)。
    至少保留 1 条（源码里 Math.max(1, keepRecent)：清光会让模型没有任何工作上下文）。"""
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
    "请把下面这段 agent 执行历史压缩成一份工作摘要，供后续继续执行任务使用。"
    "必须保留：用户的原始任务、已经完成的步骤和结论、修改过的文件、遇到的错误及原因、尚未完成的事项。"
    "不要编造，不要输出工具调用标签。\n\n"
)


class ContextManager:
    def __init__(self, llm, max_context_tokens: int = 24000, compact_ratio: Optional[float] = None,
                 keep_recent: int = 6, tool_result_max_chars: int = 6000, enabled: bool = True,
                 mc_keep_recent: int = 5):
        self.llm = llm
        self.max_context_tokens = max_context_tokens
        # 阈值 = 预算 − 缓冲（源码：有效窗口 − 13,000）；预算小时缓冲按 25% 封顶。compact_ratio 仅用于测试 / 对照实验覆盖
        self.compact_ratio = compact_ratio
        self.mc_keep_recent = mc_keep_recent
        self.keep_recent = keep_recent
        self.tool_result_max_chars = tool_result_max_chars
        self.enabled = enabled
        self.compactions = 0

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
                transcript.append(f"[{role}] {truncate_middle(content, 3000)}")
        req = [{"role": "system", "content": "你是一个负责压缩上下文的助手。"},
               {"role": "user", "content": SUMMARY_PROMPT + "\n".join(transcript)}]
        res = self.llm.chat(req, None, call_prefix="summary")
        # 摘要里如果带了工具调用标签，去掉，避免被当成调用执行
        return re.sub(r"</?tool_call>", "", res.content).strip()

    def compact(self, messages: list[dict], keep: int, reason: str) -> Optional[list[dict]]:
        if len(messages) <= keep + 2:
            return None
        cut = self._safe_boundary(messages, keep)
        old, recent = messages[1:cut], messages[cut:]
        if not old:
            return None
        summary = self._summarize(old)
        self.compactions += 1
        note = {"role": "user", "content": f"[上下文已压缩（{reason}），以下是此前工作的摘要]\n{summary}"}
        return [messages[0], note] + recent

    @property
    def threshold(self) -> int:
        if self.compact_ratio is not None:
            return int(self.max_context_tokens * self.compact_ratio)
        return self.max_context_tokens - min(AUTOCOMPACT_BUFFER_TOKENS, self.max_context_tokens // 4)

    def maybe_autocompact(self, messages: list[dict]) -> Optional[tuple[list[dict], str]]:
        """返回 (新消息, 原因) 或 None。先试 microcompact（不调模型）；还超阈值才让模型做摘要。"""
        if not self.enabled or estimate_tokens(messages) < self.threshold:
            return None
        cleared, n = clear_old_results(messages, self.mc_keep_recent)
        if n and estimate_tokens(cleared) < self.threshold:
            return cleared, "microcompact"
        summarized = self.compact(cleared, self.keep_recent, "autocompact")
        if summarized is not None:
            return summarized, "autocompact"
        return (cleared, "microcompact") if n else None

    def reactive_compact(self, messages: list[dict]) -> Optional[list[dict]]:
        return self.compact(messages, max(2, self.keep_recent // 2), "reactive")
