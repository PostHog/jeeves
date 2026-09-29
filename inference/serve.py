from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from inference.api import MODEL_ID, parse_request, response
from inference.engine import Engine
from inference.types import Options
from prep.format import DataFormat, Question

RELEASE_DATE = "2026-09-29"


class Server:
    def __init__(self, engine: Engine, defaults: Options, info: dict):
        self.engine, self.defaults, self.info = engine, defaults, info
        self.lock = threading.Lock()

    def systemone(self, body) -> dict:
        record, model, opts = parse_request(body, self.defaults)
        with self.lock:
            torch.cuda.synchronize()
            t0 = time.time()
            results = self.engine.answer(record, opts)
            torch.cuda.synchronize()
            ms = (time.time() - t0) * 1000
        return response(record, model, opts, results, self.engine.encoder.tok, ms)

    def models(self) -> dict:
        extra = {**self.info, "temperature": self.engine.head.temperature, "defaults": vars(self.defaults)}
        return {"models": [{"name": name, "description": "Jeeves-9B: reasons before it decides, with calibrated probabilities.",
                            "release_date": RELEASE_DATE, **extra} for name in (MODEL_ID, "jev-latest")]}


def handler(server: Server):
    class Handler(BaseHTTPRequestHandler):
        def send(self, code: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.send_header("x-typesafe-request-id", uuid.uuid4().hex)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path.rstrip("/") == "/v1/models":
                self.send(200, server.models())
            else:
                self.send(404, {"detail": "not found"})

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/v1/systemone":
                self.send(404, {"detail": "not found"})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"null")
                self.send(200, server.systemone(body))
            except (ValueError, KeyError) as e:
                self.send(422, {"detail": str(e)})

        def log_message(self, fmt: str, *args) -> None:
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve the Jev-compatible /v1/systemone API with speculative thinking.")
    ap.add_argument("--model", default="runs/fused")
    ap.add_argument("--drafter", default="runs/orthrus_k4/orthrus.safetensors")
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--fp8", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-rows", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--max-think", type=int, default=2560)
    ap.add_argument("--nothink-threshold", type=float, default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8009)
    a = ap.parse_args()
    engine = Engine(a.model, a.drafter, block=a.block, fp8=a.fp8, max_rows=a.max_rows, max_len=a.max_len)
    defaults = Options(max_think=a.max_think, nothink_threshold=a.nothink_threshold)
    warm = DataFormat(state="warmup", questions=[Question(id="q", type="noul", instructions="Is this a warmup?")])
    engine.answer(warm, Options(max_think=16))
    info = {"model": a.model, "drafter": a.drafter, "block": a.block, "fp8": engine.fp8}
    httpd = ThreadingHTTPServer((a.host, a.port), handler(Server(engine, defaults, info)))
    print(json.dumps({"serving": f"http://{a.host}:{a.port}", **info}), flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
