#!/usr/bin/env python3
"""Mock OpenAI-compatible server for testing the provider pipeline for free.
Streams reasoning + content, reports usage, and echoes a configurable `model` in every response
(MOCK_SERVED_MODEL, default: the requested model) so the --expect-model check can be exercised.
  python mock_server.py 18765
"""
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CODE = "```python\na, b = map(int, input().split())\nprint(a + b)\n```"


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        self._send(200, {"data": [{"id": "mock"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        model = os.environ.get("MOCK_SERVED_MODEL") or req.get("model")
        pieces = ["think "] * 3 + [CODE[i:i + 8] for i in range(0, len(CODE), 8)]
        usage = {"prompt_tokens": len(json.dumps(req["messages"])) // 4, "completion_tokens": len(pieces)}
        if not req.get("stream"):
            return self._send(200, {"model": model, "choices": [{"message": {"content": CODE}}], "usage": usage})
        self.send_response(200); self.send_header("Content-Type", "text/event-stream")
        self.send_header("x-request-id", f"mock-{time.time_ns()}"); self.send_header("Connection", "close"); self.end_headers()
        time.sleep(0.02)
        for i, p in enumerate(pieces):
            delta = {"reasoning_content": p} if i < 3 else {"content": p}
            self.wfile.write(f"data: {json.dumps({'model': model, 'choices': [{'delta': delta}]})}\n\n".encode()); self.wfile.flush()
            time.sleep(0.003)
        self.wfile.write(f"data: {json.dumps({'model': model, 'choices': [], 'usage': usage})}\n\ndata: [DONE]\n\n".encode())
        self.wfile.flush(); self.close_connection = True


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 18765), H).serve_forever()
