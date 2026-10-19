import os
import time
from pathlib import Path

import pytest

from agentrt.fs_tools import fs_tools
from agentrt.permissions import PermissionEngine
from agentrt.tools import ToolContext

T = {t.name: t for t in fs_tools()}


@pytest.fixture
def ctx(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/a.txt").write_text("hello\n")
    (tmp_path / "docs/a_copy.txt").write_text("hello\n")
    (tmp_path / "docs/Report.PDF").write_bytes(b"x" * 5000)
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules/skip.txt").write_text("hello\n")
    old = tmp_path / "docs/old.log"
    old.write_text("old")
    os.utime(old, (time.time() - 90 * 86400,) * 2)
    return ToolContext(tmp_path)


def test_find_files_by_name_and_age(ctx):
    assert "docs/Report.PDF" in T["find_files"].run({"name": "*.pdf"}, ctx)
    assert "a_copy.txt" in T["find_files"].run({"name": "a_"}, ctx)           # 子串匹配
    out = T["find_files"].run({"older_than_days": 30}, ctx)
    assert "old.log" in out and "a.txt" not in out
    assert "node_modules" not in T["find_files"].run({}, ctx)


def test_find_duplicates_includes_small_files(ctx):
    out = T["find_duplicates"].run({}, ctx)
    assert "a.txt" in out and "a_copy.txt" in out and "node_modules" not in out


def test_move_does_not_overwrite(ctx):
    with pytest.raises(Exception, match="已存在"):
        T["move_file"].run({"src": "docs/a.txt", "dst": "docs/a_copy.txt"}, ctx)
    T["move_file"].run({"src": "docs/a.txt", "dst": "archive/a.txt"}, ctx)
    assert (ctx.workspace / "archive/a.txt").exists()


def test_trash_is_recoverable(ctx):
    out = T["trash"].run({"paths": ["docs/a_copy.txt", "docs/ghost.txt"]}, ctx)
    assert not (ctx.workspace / "docs/a_copy.txt").exists()
    assert (Path(os.environ["AGENTRT_TRASH_DIR"]) / "a_copy.txt").exists()   # 进的是“废纸篓”，不是工作区里的 .trash
    assert not (ctx.workspace / ".trash").exists()
    assert "docs/a_copy.txt" in out and "ghost" in out                  # 说明原路径；不存在的跳过
    assert T["trash"].validate({"path": "."}, ctx)                      # 不能 trash 整个工作目录


def test_trash_name_collision(ctx):
    (ctx.workspace / "x").mkdir()
    (ctx.workspace / "x/a.txt").write_text("other")
    T["trash"].run({"paths": ["docs/a.txt", "x/a.txt"]}, ctx)
    assert len(list(Path(os.environ["AGENTRT_TRASH_DIR"]).iterdir())) == 2  # 同名不覆盖


def test_paths_cannot_escape(ctx):
    for name, args in [("move_file", {"src": "docs/a.txt", "dst": "../x"}),
                       ("copy_file", {"src": "../etc", "dst": "x"}),
                       ("trash", {"path": "../x"}), ("find_files", {"path": ".."})]:
        assert "越界" in T[name].validate(args, ctx)


def test_permissions_default_mode(ctx):
    eng = PermissionEngine("default")
    assert eng.decide(T["find_files"], {}, ctx).behavior == "allow"
    assert eng.decide(T["trash"], {"path": "docs/a.txt"}, ctx).behavior == "ask"
    assert PermissionEngine("plan").decide(T["move_file"], {"src": "a", "dst": "b"}, ctx).behavior == "deny"
