#!/usr/bin/env python3
"""
Compliant Agents viewer server.

Usage:
    python viewer/server.py
    open http://localhost:8765
"""

from http.server import ThreadingHTTPServer

from server_app.config import DATA, HOST, PORT
from server_app.experiments import _ensure_worker
from server_app.http import Handler
from server_app.mapper import _ensure_mapper_worker


def main() -> None:
    _ensure_worker()
    _ensure_mapper_worker()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print("\n  Compliant Agents Viewer")
    print(f"  -> http://{HOST}:{PORT}")
    print(f"  -> data directory: {DATA}")
    print("  -> Ctrl+C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")


if __name__ == "__main__":
    main()
