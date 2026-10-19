"""测试用的最小 MCP server（stdio，换行分隔的 JSON-RPC）。"""
import json
import sys

TOOLS = [
    {"name": "echo", "description": "回显", "annotations": {"readOnlyHint": True},
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
    {"name": "write_note", "description": "写笔记（会改动数据）",
     "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "boom", "description": "总是报错", "annotations": {"readOnlyHint": True},
     "inputSchema": {"type": "object", "properties": {}}},
]
for line in sys.stdin:
    msg = json.loads(line)
    if "id" not in msg:
        continue
    m, p = msg["method"], msg.get("params", {})
    if m == "initialize":
        res = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "fake"}}
    elif m == "tools/list":
        res = {"tools": TOOLS}
    elif m == "tools/call":
        print("这行不是 JSON，客户端应忽略", flush=True)
        if p["name"] == "boom":
            res = {"content": [{"type": "text", "text": "出错了"}], "isError": True}
        else:
            res = {"content": [{"type": "text", "text": f"{p['name']}:{json.dumps(p['arguments'], ensure_ascii=False)}"}]}
    else:
        print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": "no such method"}}), flush=True)
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": res}), flush=True)
