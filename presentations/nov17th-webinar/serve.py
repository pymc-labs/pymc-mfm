#!/usr/bin/env python3
"""Serve the webinar and its shared public fonts on loopback, without dependencies."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

DECK = Path(__file__).resolve().parent
SHARED = DECK.parents[1] / "docs" / "assets"


class PresentationHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        url_path = unquote(urlsplit(path).path)
        for folder in ("fonts", "favicon"):
            prefix = f"/docs/assets/{folder}/"
            if url_path.startswith(prefix):
                root = SHARED / folder
                candidate = (root / url_path.removeprefix(prefix)).resolve()
                if candidate.is_relative_to(root) and candidate.is_file():
                    return str(candidate)
                return str(DECK / ".not-found")
        return super().translate_path(path)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8717)
    args = parser.parse_args()
    handler = partial(PresentationHandler, directory=str(DECK))
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"Webinar: http://127.0.0.1:{server.server_port}", flush=True)
        print("P: presenter screen. F: full screen. Arrow keys: navigate.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
