"""文件搜索 / 整理工具（assistant 用）。

和 tools.py 里的工具一样受工作目录约束（validate 与 run 两处检查）。
没有“永久删除”：trash 把文件放进系统废纸篓（macOS 走访达，可「放回原处」）。
"""
from __future__ import annotations

import fnmatch
import hashlib
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Optional

from .tools import Tool, ToolContext, ToolError, _obj, resolve_in_workspace

SKIP_DIRS = {".git", ".agentrt", ".trash", "node_modules", "__pycache__", ".venv", "venv", ".Trash",
             "Library", ".cache", ".DS_Store"}
MAX_HITS = 200


def walk(root: Path, include_hidden: bool = False) -> Iterator[Path]:
    """递归遍历文件，跳过噪音目录；不跟随符号链接。"""
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return
    for p in entries:
        if p.name in SKIP_DIRS or (p.name.startswith(".") and not include_hidden):
            continue
        if p.is_symlink():
            continue
        if p.is_dir():
            yield from walk(p, include_hidden)
        elif p.is_file():
            yield p


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"


def _rel(ctx: ToolContext, p: Path) -> str:
    return str(p.relative_to(ctx.workspace.resolve())) or "."


class _PathsTool(Tool):
    """validate 同时检查 path / src / dst。"""
    path_keys = ("path", "src", "dst")

    def validate(self, args, ctx):
        for key in self.parameters.get("required", []):
            if key not in args:
                return f"缺少必填参数 {key}"
        for k in self.path_keys:
            if k in args:
                try:
                    resolve_in_workspace(ctx, str(args[k]))
                except ToolError as e:
                    return str(e)
        return None


_FILTER_PROPS = {"path": {"type": "string", "description": "起始目录，默认 ."},
                 "name": {"type": "string", "description": "通配符如 *.pdf（不区分大小写）；无通配符时按子串匹配"},
                 "min_size_mb": {"type": "number"}, "max_size_mb": {"type": "number"},
                 "modified_after": {"type": "string", "description": "YYYY-MM-DD，含当天"},
                 "modified_before": {"type": "string", "description": "YYYY-MM-DD，含当天"},
                 "modified_within_days": {"type": "number"}, "older_than_days": {"type": "number"},
                 "include_hidden": {"type": "boolean"}}


def _day(s: str, end: bool) -> float:
    try:
        t = time.mktime(time.strptime(str(s), "%Y-%m-%d"))
    except ValueError:
        raise ToolError(f"日期格式应为 YYYY-MM-DD：{s}") from None
    return t + 86400 - 1 if end else t


def select(args: dict, ctx: ToolContext) -> list[tuple[Path, int, float]]:
    """按 _FILTER_PROPS 里的条件筛文件，返回 (路径, 大小, 修改时间)。"""
    root = resolve_in_workspace(ctx, args.get("path", "."))
    if not root.is_dir():
        raise ToolError(f"{args.get('path', '.')} 不是目录")
    pat = str(args.get("name") or "").lower()
    if pat and not any(c in pat for c in "*?["):
        pat = f"*{pat}*"
    now = time.time()
    lo, hi = 0.0, float("inf")
    if args.get("modified_after"):
        lo = _day(args["modified_after"], False)
    if args.get("modified_before"):
        hi = _day(args["modified_before"], True)
    if args.get("modified_within_days"):
        lo = max(lo, now - float(args["modified_within_days"]) * 86400)
    if args.get("older_than_days"):
        hi = min(hi, now - float(args["older_than_days"]) * 86400)
    min_size = float(args.get("min_size_mb") or 0) * 1024 * 1024
    max_size = float(args["max_size_mb"]) * 1024 * 1024 if args.get("max_size_mb") else float("inf")
    rows = []
    for f in walk(root, bool(args.get("include_hidden"))):
        if pat and not fnmatch.fnmatch(f.name.lower(), pat):
            continue
        try:
            st = f.stat()
        except OSError:
            continue
        if min_size <= st.st_size <= max_size and lo <= st.st_mtime <= hi:
            rows.append((f, st.st_size, st.st_mtime))
    return rows


class FindFiles(_PathsTool):
    name = "find_files"
    description = "按名称 / 大小 / 修改日期找文件，sort 可选 name / size / mtime（新的在前）。最多列 200 条。"
    parameters = _obj({**_FILTER_PROPS, "sort": {"type": "string"}}, [])

    def run(self, args, ctx):
        rows = select(args, ctx)
        key = {"size": lambda r: -r[1], "mtime": lambda r: -r[2]}.get(args.get("sort"), lambda r: str(r[0]))
        rows.sort(key=key)
        lines = [f"{_rel(ctx, f)}  {human(sz)}  {time.strftime('%Y-%m-%d', time.localtime(mt))}"
                 for f, sz, mt in rows[:MAX_HITS]]
        head = f"共 {len(rows)} 个文件，合计 {human(sum(r[1] for r in rows))}\n"
        if len(rows) > MAX_HITS:
            lines.append(f"[只显示前 {MAX_HITS} 个；请收窄条件]")
        return head + "\n".join(lines) if rows else "（没有符合条件的文件）"


class FileStats(_PathsTool):
    name = "file_stats"
    description = ("统计文件数量和占用空间，按 group_by 分组：ext（扩展名）/ month（修改月份）/ year / folder（一级子目录）。"
                   "可带 find_files 的同样过滤条件。")
    parameters = _obj({**_FILTER_PROPS, "group_by": {"type": "string", "description": "ext / month / year / folder，默认 ext"}}, [])

    def run(self, args, ctx):
        rows = select(args, ctx)
        root = resolve_in_workspace(ctx, args.get("path", "."))
        mode = args.get("group_by", "ext")
        groups: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for f, sz, mt in rows:
            if mode == "month":
                k = time.strftime("%Y-%m", time.localtime(mt))
            elif mode == "year":
                k = time.strftime("%Y", time.localtime(mt))
            elif mode == "folder":
                parts = f.relative_to(root).parts
                k = parts[0] if len(parts) > 1 else "(根目录)"
            else:
                k = f.suffix.lower() or "(无扩展名)"
            groups[k][0] += 1
            groups[k][1] += sz
        if not rows:
            return "（没有符合条件的文件）"
        order = sorted(groups.items(), key=(lambda kv: kv[0]) if mode in ("month", "year") else (lambda kv: -kv[1][1]))
        out = [f"共 {len(rows)} 个文件，合计 {human(sum(r[1] for r in rows))}", f"{'分组':<16}{'数量':>8}{'大小':>12}"]
        out += [f"{k:<16}{n:>8}{human(sz):>12}" for k, (n, sz) in order[:60]]
        return "\n".join(out)


class DiskUsage(_PathsTool):
    name = "disk_usage"
    description = "统计目录下各一级子项占用的空间（从大到小），用来找出占地方的东西。"
    parameters = _obj({"path": {"type": "string", "description": "默认 ."},
                       "top": {"type": "integer", "description": "显示前几项，默认 15"}}, [])

    def run(self, args, ctx):
        root = resolve_in_workspace(ctx, args.get("path", "."))
        if not root.is_dir():
            raise ToolError(f"{args.get('path', '.')} 不是目录")
        sizes: dict[str, int] = defaultdict(int)
        total = 0
        for f in walk(root, include_hidden=True):
            try:
                sz = f.stat().st_size
            except OSError:
                continue
            sizes[f.relative_to(root).parts[0]] += sz
            total += sz
        top = sorted(sizes.items(), key=lambda kv: -kv[1])[:int(args.get("top", 15))]
        out = [f"{human(sz):>10}  {name}" for name, sz in top]
        return f"合计 {human(total)}（已跳过 .git / node_modules / 符号链接等）\n" + "\n".join(out)


class FindDuplicates(_PathsTool):
    name = "find_duplicates"
    description = "找内容完全相同的重复文件（先按大小分组再比对 SHA-256）。只报告，不删除。"
    parameters = _obj({"path": {"type": "string", "description": "默认 ."},
                       "min_size_kb": {"type": "number", "description": "忽略更小的文件，默认 0（只忽略空文件）"}}, [])

    def run(self, args, ctx):
        root = resolve_in_workspace(ctx, args.get("path", "."))
        min_size = max(1.0, float(args.get("min_size_kb", 0)) * 1024)
        by_size: dict[int, list[Path]] = defaultdict(list)
        for f in walk(root):
            try:
                sz = f.stat().st_size
            except OSError:
                continue
            if sz >= min_size:
                by_size[sz].append(f)
        groups = []
        for sz, files in by_size.items():
            if len(files) < 2:
                continue
            by_hash: dict[str, list[Path]] = defaultdict(list)
            for f in files:
                h = hashlib.sha256()
                try:
                    with open(f, "rb") as fh:
                        for chunk in iter(lambda: fh.read(1 << 20), b""):
                            h.update(chunk)
                except OSError:
                    continue
                by_hash[h.hexdigest()].append(f)
            groups += [(sz, fs) for fs in by_hash.values() if len(fs) > 1]
        groups.sort(key=lambda g: -g[0] * (len(g[1]) - 1))
        if not groups:
            return "（没有重复文件）"
        wasted = sum(sz * (len(fs) - 1) for sz, fs in groups)
        out = [f"{len(groups)} 组重复，可回收约 {human(wasted)}"]
        for sz, fs in groups[:30]:
            out.append(f"\n[{human(sz)} × {len(fs)}]")
            out += [f"  {_rel(ctx, f)}" for f in sorted(fs)]
        return "\n".join(out)


class MakeDir(_PathsTool):
    name = "make_dir"
    description = "创建目录（含父目录）。"
    parameters = _obj({"path": {"type": "string"}}, ["path"])
    read_only = False
    risk = "medium"

    def run(self, args, ctx):
        resolve_in_workspace(ctx, args["path"]).mkdir(parents=True, exist_ok=True)
        return f"已创建目录 {args['path']}"


class MoveFile(_PathsTool):
    name = "move_file"
    description = "移动或重命名文件/目录：src → dst。dst 已存在会报错，不会覆盖；dst 是已有目录时移入其中；dst 的父目录不存在会自动创建（所以不需要单独建目录）。"
    parameters = _obj({"src": {"type": "string"}, "dst": {"type": "string"}}, ["src", "dst"])
    read_only = False
    risk = "medium"

    def subject(self, args):
        return str(args.get("src", ""))

    def run(self, args, ctx):
        src = resolve_in_workspace(ctx, args["src"])
        dst = resolve_in_workspace(ctx, args["dst"])
        if not src.exists():
            raise ToolError(f"源不存在：{args['src']}")
        if dst.is_dir():
            dst = dst / src.name
        if dst.exists():
            raise ToolError(f"目标已存在：{_rel(ctx, dst)}（不会覆盖）")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return f"已移动 {args['src']} → {_rel(ctx, dst)}"


class CopyFile(_PathsTool):
    name = "copy_file"
    description = "复制文件/目录：src → dst。dst 已存在会报错，不会覆盖。"
    parameters = _obj({"src": {"type": "string"}, "dst": {"type": "string"}}, ["src", "dst"])
    read_only = False
    risk = "medium"

    def subject(self, args):
        return str(args.get("src", ""))

    def run(self, args, ctx):
        src = resolve_in_workspace(ctx, args["src"])
        dst = resolve_in_workspace(ctx, args["dst"])
        if not src.exists():
            raise ToolError(f"源不存在：{args['src']}")
        if dst.is_dir():
            dst = dst / src.name
        if dst.exists():
            raise ToolError(f"目标已存在：{_rel(ctx, dst)}（不会覆盖）")
        dst.parent.mkdir(parents=True, exist_ok=True)
        (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst)
        return f"已复制 {args['src']} → {_rel(ctx, dst)}"


def _unique(dest_dir: Path, name: str) -> Path:
    d = dest_dir / name
    n = 1
    while d.exists() or d.is_symlink():
        d = dest_dir / f"{Path(name).stem} {time.strftime('%H.%M.%S')}{'' if n == 1 else f' {n}'}{Path(name).suffix}"
        n += 1
    return d


def _finder_trash(paths: list[Path]) -> bool:
    """交给访达删除：文件进系统废纸篓，并保留「放回原处」信息。需要首次授权“自动化”，失败返回 False。"""
    items = ", ".join('POSIX file "%s"' % str(p).replace("\\", "\\\\").replace('"', '\\"') for p in paths)
    try:
        r = subprocess.run(["osascript", "-e", f'tell application "Finder" to delete {{{items}}}'],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0 and not any(p.exists() for p in paths)


def _move_to_trash_dir(paths: list[Path], trash_dir: Path) -> None:
    trash_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        shutil.move(str(p), str(_unique(trash_dir, p.name)))


def system_trash(paths: list[Path]) -> str:
    """把文件放进系统废纸篓，返回说明。AGENTRT_TRASH_DIR 可以改成指定目录（测试 / 非 macOS 自定义）。"""
    override = os.environ.get("AGENTRT_TRASH_DIR")
    if override:
        _move_to_trash_dir(paths, Path(override).expanduser())
        return f"已移入 {override}"
    if sys.platform == "darwin":
        if _finder_trash(paths):
            return "已移入系统废纸篓（在访达里右键 → 放回原处 即可恢复）"
        _move_to_trash_dir([p for p in paths if p.exists()], Path.home() / ".Trash")
        return "已移入 ~/.Trash（访达授权不可用，没有“放回原处”信息，请按下面列出的原路径手动移回）"
    xdg = Path.home() / ".local/share/Trash/files"
    _move_to_trash_dir(paths, xdg)
    return "已移入系统回收站 ~/.local/share/Trash/files"


class Trash(_PathsTool):
    name = "trash"
    description = ("“删除”文件/目录：放进系统废纸篓（macOS 访达废纸篓，可「放回原处」），没有永久删除。"
                   "path 删一个；paths 一次删多个（用户会在确认框里看到完整清单）。")
    parameters = _obj({"path": {"type": "string"},
                       "paths": {"type": "array", "items": {"type": "string"}}}, [])
    read_only = False
    risk = "high"  # auto 模式下也要确认

    @staticmethod
    def _targets(args) -> list[str]:
        return ([args["path"]] if args.get("path") else []) + [str(p) for p in args.get("paths") or []]

    def subject(self, args):
        return ",".join(self._targets(args))

    def validate(self, args, ctx):
        targets = self._targets(args)
        if not targets:
            return "需要 path 或 paths"
        for t in targets:
            try:
                p = resolve_in_workspace(ctx, t)
            except ToolError as e:
                return str(e)
            if p == ctx.workspace.resolve():
                return "不能对工作目录本身执行 trash"
        return None

    def run(self, args, ctx):
        found, missing = [], []
        for t in self._targets(args):
            (found if resolve_in_workspace(ctx, t).exists() else missing).append(t)
        if found:
            note = system_trash([resolve_in_workspace(ctx, t) for t in found])
            msg = f"{note}：{len(found)} 项\n原路径：" + "；".join(found[:50]) + ("…" if len(found) > 50 else "")
        else:
            msg = "没有可删除的项"
        if missing:
            msg += f"\n不存在，已跳过：{', '.join(missing[:10])}"
        return msg


def fs_tools() -> list[Tool]:
    return [FindFiles(), FileStats(), DiskUsage(), FindDuplicates(), MakeDir(), MoveFile(), CopyFile(), Trash()]
