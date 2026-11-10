"""会话持久化：只追加的 JSONL 事件日志（event sourcing）。

每一步（模型输出、工具开始、工具结果、压缩、结束）都先落盘再继续，所以进程崩溃后
可以 replay 日志恢复上下文，从断点继续。

压缩不改写已有事件，只追加 compact。模型看到的消息是 project_context 算出来的投影：
最新一条摘要 compact 之前的历史用摘要代替，microcompact 把指定工具结果换成占位符。
read-before-write 的内容哈希是 file_state 事件，不进消息列表，压缩丢不掉，resume 时按路径装回。

恢复时对「开始了但没有结果」的工具调用区分语义：
- 只读工具：重新执行（至少一次，重复执行无副作用）
- 有副作用的工具：不重跑（至多一次），把「结果未知」告诉模型，让它先检查状态再决定
写日志前会对疑似凭据做脱敏。
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Optional

_REDACT = [
    (re.compile(r"(sshpass\s+-p\s*)\S+"), r"\1***"),
    (re.compile(r"(-p')[^']+(')"), r"\1***\2"),
    (re.compile(r"\b(sk-|tvly-)[A-Za-z0-9\-]{12,}"), r"\1***"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AKIA***"),
    (re.compile(r"(?i)\b(password|passwd|token|secret)(\s*=\s*)\S+"), r"\1\2***"),
]


def redact(obj: Any) -> Any:
    if isinstance(obj, str):
        for rx, rep in _REDACT:
            obj = rx.sub(rep, obj)
        return obj
    if isinstance(obj, list):
        return [redact(x) for x in obj]
    if isinstance(obj, dict):
        return {k: redact(v) for k, v in obj.items()}
    return obj


class EventLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.seq = 0
        if self.path.exists():
            self.seq = sum(1 for _ in self.path.open())

    @classmethod
    def new(cls, sessions_dir: Path, session_id: Optional[str] = None) -> "EventLog":
        sid = session_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        return cls(Path(sessions_dir) / f"{sid}.jsonl")

    @property
    def session_id(self) -> str:
        return self.path.stem

    def append(self, event: dict) -> dict:
        self.seq += 1
        rec = {"seq": self.seq, "ts": round(time.time(), 3), **redact(event)}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return rec

    def events(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                break  # 最后一行可能写了一半：丢弃，之前的都是完整的
        return out


def project_context(raw: list[dict], compacts: list[dict]) -> list[dict]:
    """模型视图。raw 是按日志顺序排好、带 id（事件 seq）的消息；compacts 是 compact 事件。

    最新一条带 summary 的 compact 决定切点：system 留下，切点之前用摘要代替，切点之后（含 compact
    之后新写下的消息）原样保留。retain_none 时切点之前一条不留。microcompact 的 cleared_ids 把
    还留在视图里的工具结果换成占位符；原文仍在 tool_result 事件里，下次摘要读得到。
    """
    from .context import CLEARED, render_summary

    cleared: list[str] = []
    summary_event = None
    for e in compacts:
        if e.get("cleared_ids"):
            cleared.extend(e["cleared_ids"])
        if "summary" in e:
            summary_event = e  # 后一次摘要覆盖前一次；文件清单已经累加在这条事件上
    clear = set(cleared)

    def scrub(m: dict) -> dict:
        if m.get("role") == "tool" and m.get("tool_call_id") in clear and m.get("content") != CLEARED:
            return {**m, "content": CLEARED}
        return m

    if summary_event is None:
        return [scrub(m) for m in raw]
    system = [raw[0]] if raw and raw[0].get("role") == "system" else []
    rest = raw[1:] if system else raw
    kept_id = summary_event.get("first_kept_id")
    retain_none = bool(summary_event.get("retain_none"))
    compact_seq = summary_event.get("seq") or 0
    kept = []
    for m in rest:
        mid = m.get("id")
        if mid is None:
            continue
        if retain_none:
            if mid > compact_seq:
                kept.append(m)
        elif kept_id is None:
            kept.append(m)
        elif mid >= kept_id:
            kept.append(m)
    note = {"role": "user", "kind": "summary",
            "content": render_summary(summary_event.get("reason") or "compact", summary_event.get("summary") or "",
                                      list(summary_event.get("read_files") or []),
                                      list(summary_event.get("modified_files") or []))}
    return [scrub(m) for m in system + [note] + kept]


def _legacy_messages(events: list[dict]) -> list[dict]:
    """旧日志：compact 事件里嵌着整份新消息列表，读到它就整表替换。新日志不再这样写。"""
    messages: list[dict] = []
    for e in events:
        t = e.get("type")
        if t == "message":
            m = dict(e["message"])
            m.setdefault("id", e.get("seq"))
            messages.append(m)
        elif t == "tool_result":
            messages.append({"role": "tool", "id": e.get("seq"), "tool_call_id": e["call_id"],
                             "name": e["name"], "content": e["content"]})
        elif t == "compact" and "messages" in e:
            messages = [dict(m) for m in e["messages"]]
    return messages


class ReplayState:
    """从事件日志重建的状态。messages 是投影，不是日志的逐条拷贝。"""

    def __init__(self, events: list[dict]):
        self.messages: list[dict] = []
        self.started: dict[str, dict] = {}   # call_id -> tool_start 事件
        self.finished: set[str] = set()
        self.procs: dict[str, dict] = {}     # call_id -> tool_proc 事件（进程组号、启动时间）
        self.file_digests: dict[str, str] = {}  # 路径 → 上次读到或写下的内容哈希
        self.ended: Optional[dict] = None
        self.turns = 0
        raw: list[dict] = []
        compacts: list[dict] = []
        legacy = False
        for e in events:
            t = e.get("type")
            if t == "message":
                m = dict(e["message"])
                m.setdefault("id", e.get("seq"))
                raw.append(m)
                if m["role"] == "assistant":
                    self.turns += 1
            elif t == "dropped_reply":
                self.turns += 1   # 没进上下文的回复也用掉了一个调用序号；call id 按序号生成，恢复后不能重复
            elif t == "tool_start":
                self.started[e["call_id"]] = e
            elif t == "tool_proc":
                self.procs[e["call_id"]] = e
            elif t == "tool_result":
                self.finished.add(e["call_id"])
                raw.append({"role": "tool", "id": e.get("seq"), "tool_call_id": e["call_id"],
                            "name": e["name"], "content": e["content"]})
            elif t == "compact":
                compacts.append(e)
                if "messages" in e and "summary" not in e:
                    legacy = True
            elif t == "file_state":
                self.file_digests[e["path"]] = e["digest"]
            elif t == "turn_start":
                self.ended = None   # 多轮对话：新一轮开始后，前一轮的 end 不再代表“已结束”
            elif t == "end":
                self.ended = e
        self.messages = _legacy_messages(events) if legacy else project_context(raw, compacts)

    def pending_calls(self) -> list[dict]:
        """最后一条 assistant 消息里尚未产出结果的工具调用。"""
        for m in reversed(self.messages):
            if m["role"] == "assistant":
                return [c for c in (m.get("tool_calls") or []) if c["id"] not in self.finished]
            if m["role"] == "user":
                return []
        return []
