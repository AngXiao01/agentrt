"""会话持久化：只追加的 JSONL 事件日志（event sourcing）。

每一步（模型输出、工具开始、工具结果、压缩、结束）都先落盘再继续，所以进程崩溃后
可以 replay 日志恢复上下文，从断点继续。

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


class ReplayState:
    """从事件日志重建的状态。"""

    def __init__(self, events: list[dict]):
        self.messages: list[dict] = []
        self.started: dict[str, dict] = {}   # call_id -> tool_start 事件
        self.finished: set[str] = set()
        self.ended: Optional[dict] = None
        self.turns = 0
        for e in events:
            t = e.get("type")
            if t == "message":
                self.messages.append(e["message"])
                if e["message"]["role"] == "assistant":
                    self.turns += 1
            elif t == "tool_start":
                self.started[e["call_id"]] = e
            elif t == "tool_result":
                self.finished.add(e["call_id"])
                self.messages.append({"role": "tool", "tool_call_id": e["call_id"], "name": e["name"],
                                      "content": e["content"]})
            elif t == "compact":
                self.messages = e["messages"]
            elif t == "turn_start":
                self.ended = None   # 多轮对话：新一轮开始后，前一轮的 end 不再代表“已结束”
            elif t == "end":
                self.ended = e

    def pending_calls(self) -> list[dict]:
        """最后一条 assistant 消息里尚未产出结果的工具调用。"""
        for m in reversed(self.messages):
            if m["role"] == "assistant":
                return [c for c in (m.get("tool_calls") or []) if c["id"] not in self.finished]
            if m["role"] == "user":
                return []
        return []
