"""端到端：CLI → HTTP → 假 vLLM → 工具 → 日志 → trace。"""
import json

from agentrt.cli import main

from conftest import call, completion


def test_cli_run_and_trace(fake_vllm, tmp_path, capsys):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "n.txt").write_text("1\n2\n3\n")
    fake_vllm.queue = [(200, completion(call("read_file", path="n.txt"), prompt_tokens=500, cached=0)),
                       (200, completion(call("write_file", path="sum.txt", content="6"), prompt_tokens=600, cached=480)),
                       (200, completion("和是 6，已写入 sum.txt。", prompt_tokens=700, cached=580))]
    sessions = tmp_path / "sessions"
    main(["--sessions-dir", str(sessions), "run", "把 n.txt 里的数加起来写到 sum.txt", "--workspace", str(ws),
          "--base-url", fake_vllm.url, "--mode", "auto", "--approve", "deny"])
    out = capsys.readouterr().out
    assert "和是 6" in out and "命中率" in out
    assert (ws / "sum.txt").read_text() == "6"
    # 发给服务端的消息：text 协议下 user/assistant 严格交替，工具结果变成 user 消息
    roles = [m["role"] for m in fake_vllm.requests[-1]["messages"]]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    sid = next(sessions.glob("*.jsonl")).stem
    main(["--sessions-dir", str(sessions), "trace", sid, "--timeline"])
    out = capsys.readouterr().out
    assert "read_file" in out and "write_file" in out and "end_turn" in out
