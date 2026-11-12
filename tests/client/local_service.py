"""A local HTTP stub of the analysis routes, for tests that need real sockets and processes.

It enforces idempotency by key the way the service does: an equal key with an equal body
returns the stored analysis (a deleted one answers 410); an equal key with another body answers
409. The worker is held, so every analysis stays ``running``. Responses are built from
``client_fixtures``, which ``test_contract.py`` checks against ``contract.json``.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .client_fixtures import EXPIRES, error_body, preview_record, started, status_record

API_KEY = "test-key"
ANALYSIS_ROUTE = "/v1/analyses/"


class StubState:
    """The analyses the stub stores, in creation order."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.by_key: dict[str, str] = {}
        self.analyses: dict[str, dict[str, Any]] = {}

    def create(self, key: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        with self.lock:
            if key in self.by_key:
                stored = self.analyses[self.by_key[key]]
                if stored["body"] != body:
                    return 409, error_body("idempotency_conflict", "key used with another body")
                if stored["deleted"]:
                    return 410, error_body("gone", "deleted")
                return 202, started() | {"analysis_id": stored["id"]}
            analysis_id = f"an_{len(self.analyses) + 1}"
            self.by_key[key] = analysis_id
            self.analyses[analysis_id] = {"id": analysis_id, "body": body, "deleted": False}
            return 202, started() | {"analysis_id": analysis_id}

    def listing(self) -> dict[str, Any]:
        with self.lock:
            return {
                "analyses": [
                    {
                        "analysis_id": item["id"],
                        "dataset_id": item["body"]["dataset_id"],
                        "name": None,
                        "url": None,
                        "status": "running",
                        "runs_total": 3,
                        "runs_done": 1,
                        "runs_failed": 0,
                        "created_at": "2029-12-25T00:00:00Z",
                        "expires_at": EXPIRES,
                    }
                    for item in self.analyses.values()
                    if not item["deleted"]
                ]
            }

    def delete(self, analysis_id: str) -> tuple[int, dict[str, Any]]:
        with self.lock:
            stored = self.analyses.get(analysis_id)
            if stored is None or stored["deleted"]:
                return 410, error_body("gone", "deleted")
            stored["deleted"] = True
            return 202, {}

    def status(self, analysis_id: str) -> tuple[int, dict[str, Any]]:
        with self.lock:
            stored = self.analyses.get(analysis_id)
        if stored is None:
            return 404, error_body("not_found", analysis_id)
        if stored["deleted"]:
            return 410, error_body("gone", analysis_id)
        return 200, status_record("running")


def make_handler(state: StubState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # keep test output quiet
            pass

        def reply(self, status: int, body: dict[str, Any]) -> None:
            content = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def authorized(self) -> bool:
            if self.headers.get("Authorization") == f"Bearer {API_KEY}":
                return True
            self.reply(401, error_body("unauthorized", "bad key"))
            return False

        def do_GET(self) -> None:
            if not self.authorized():
                return
            path = self.path.split("?")[0]
            if path == "/v1/analyses":
                self.reply(200, state.listing())
            elif path.startswith(ANALYSIS_ROUTE):
                self.reply(*state.status(path.removeprefix(ANALYSIS_ROUTE)))
            else:
                self.reply(404, error_body("not_found", path))

        def do_DELETE(self) -> None:
            if self.authorized() and self.path.startswith(ANALYSIS_ROUTE):
                self.reply(*state.delete(self.path.removeprefix(ANALYSIS_ROUTE)))

        def do_POST(self) -> None:
            if not self.authorized():
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path != "/v1/analyses":
                self.reply(404, error_body("not_found", self.path))
            elif body.get("preview"):
                self.reply(200, preview_record())
            elif "Idempotency-Key" not in self.headers:
                self.reply(422, error_body("missing_field", "Idempotency-Key", "Idempotency-Key"))
            else:
                self.reply(*state.create(self.headers["Idempotency-Key"], body))

    return Handler


@contextmanager
def serve():
    """Serve the stub on a free 127.0.0.1 port; yield ``(base_url, state)``."""
    state = StubState()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
