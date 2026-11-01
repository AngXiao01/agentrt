from agentrt.llm import (ContextLengthError, OpenAICompatLLM, RetryableError, parse_text_tool_calls,
                         to_wire)

from conftest import completion


def test_parse_text_tool_calls():
    text = ('先看目录\n<tool_call>{"name": "list_dir", "arguments": {"path": "."}}</tool_call>\n'
            '<tool_call>```json\n{"name": "read_file", "arguments": "{\\"path\\": \\"a.txt\\"}"}\n```</tool_call>')
    calls, errors = parse_text_tool_calls(text, "t1")
    assert errors == []
    assert [c.name for c in calls] == ["list_dir", "read_file"]
    assert calls[1].arguments == {"path": "a.txt"}
    assert [c.id for c in calls] == ["t1_0", "t1_1"]


def test_parse_errors_are_reported_not_dropped():
    calls, errors = parse_text_tool_calls('<tool_call>{"name": "x", "arguments": {bad}}</tool_call>', "t")
    assert calls == [] and len(errors) == 1
    calls, errors = parse_text_tool_calls('<tool_call>{"name": "x"', "t")
    assert calls == [] and "闭合" in errors[0]


def test_gemma_native_call_syntax_is_a_parse_error_not_a_final_answer():
    calls, errors = parse_text_tool_calls('call:todo_write{items:["a","b"]}', "t")
    assert calls == [] and len(errors) == 1 and "<tool_call>" in errors[0] and "todo_write" in errors[0]
    calls, errors = parse_text_tool_calls("结论：已完成，不需要 recall:foo{x} 之类的调用", "t")
    assert calls == [] and errors == []


def test_to_wire_alternation_and_merge_system():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
            {"role": "assistant", "content": "A", "tool_calls": [{"id": "c0", "name": "ls", "arguments": {}}]},
            {"role": "tool", "tool_call_id": "c0", "name": "ls", "content": "R0"},
            {"role": "tool", "tool_call_id": "c1", "name": "ls", "content": "R1"},
            {"role": "user", "content": "note"}]
    w = to_wire(msgs, native_tools=False, merge_system=True)
    assert [m["role"] for m in w] == ["user", "assistant", "user"]  # 严格交替
    assert w[0]["content"].startswith("S")
    assert "R0" in w[2]["content"] and "R1" in w[2]["content"] and "note" in w[2]["content"]
    wn = to_wire(msgs, native_tools=True, merge_system=False)
    assert [m["role"] for m in wn] == ["system", "user", "assistant", "tool", "tool", "user"]
    assert wn[2]["tool_calls"][0]["function"]["name"] == "ls"


def test_retry_then_success(fake_vllm):
    fake_vllm.queue = [(503, {"error": "busy"}), (429, {"error": "rate"}), (200, completion("OK"))]
    slept = []
    llm = OpenAICompatLLM(fake_vllm.url, sleep=slept.append)
    assert llm.model == "fake-gemma"
    res = llm.chat([{"role": "user", "content": "hi"}])
    assert res.content == "OK" and len(slept) == 2
    assert fake_vllm.requests[-1]["stop"] == ["<tool_result"]


def test_retry_exhausted(fake_vllm):
    fake_vllm.queue = [(500, {})] * 3
    llm = OpenAICompatLLM(fake_vllm.url, model="fake-gemma", max_retries=2, sleep=lambda s: None)
    try:
        llm.chat([{"role": "user", "content": "hi"}])
        raise AssertionError("应当抛出 RetryableError")
    except RetryableError:
        pass


def test_context_length_error(fake_vllm):
    fake_vllm.queue = [(400, {"message": "This model's maximum context length is 8192 tokens"})]
    llm = OpenAICompatLLM(fake_vllm.url, model="fake-gemma", sleep=lambda s: None)
    try:
        llm.chat([{"role": "user", "content": "hi"}])
        raise AssertionError
    except ContextLengthError:
        pass


def test_native_tool_calls(fake_vllm):
    tc = [{"id": "call_1", "type": "function", "function": {"name": "list_dir", "arguments": '{"path": "."}'}}]
    fake_vllm.queue = [(200, completion("", tool_calls=tc, cached=64))]
    llm = OpenAICompatLLM(fake_vllm.url, model="fake-gemma", native_tools=True)
    res = llm.chat([{"role": "user", "content": "hi"}], [{"type": "function", "function": {"name": "list_dir"}}])
    assert res.tool_calls[0].id == "call_1" and res.tool_calls[0].arguments == {"path": "."}
    assert "tools" in fake_vllm.requests[-1] and "stop" not in fake_vllm.requests[-1]
    assert res.usage["prompt_tokens_details"]["cached_tokens"] == 64


def test_context_budget_follows_server(fake_vllm):
    llm = OpenAICompatLLM(fake_vllm.url, max_tokens=2048)
    assert llm.max_model_len == 8192
    assert llm.context_budget(24000) == 8192 - 2048 - 512
    assert llm.context_budget(4000) == 4000


def test_unknown_model_rejected(fake_vllm):
    try:
        OpenAICompatLLM(fake_vllm.url, model="nope")
        raise AssertionError
    except Exception as e:  # noqa: BLE001
        assert "nope" in str(e)


def test_midread_disconnect_is_retried(fake_vllm, monkeypatch):
    import http.client
    import urllib.request
    real = urllib.request.urlopen
    state = {"n": 0}

    def flaky(req, timeout=None):
        if req.full_url.endswith("/chat/completions"):
            state["n"] += 1
            if state["n"] <= 2:
                raise http.client.IncompleteRead(b"x" * 10)
        return real(req, timeout=timeout)

    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    fake_vllm.queue = [(200, completion("OK"))]
    slept = []
    llm = OpenAICompatLLM(fake_vllm.url, model="fake-gemma", sleep=slept.append, max_backoff=5.0)
    assert llm.chat([{"role": "user", "content": "hi"}]).content == "OK"
    assert len(slept) == 2 and max(slept) <= 5.0 * 1.5
