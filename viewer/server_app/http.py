from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

from .config import HERE
from .domains import get_domain_journey_options, start_domain_journey
from .experiments import (
    _cancel_experiment,
    _compact_experiment,
    _ensure_worker,
    _experiment_from_payload,
    get_experiment_options,
)
from .live_chat import (
    create_live_session,
    get_live_options,
    get_live_session,
    send_live_message,
)
from .mapper import (
    _cancel_mapper_experiment,
    _compact_mapper_experiment,
    _ensure_mapper_worker,
    _mapper_experiment_from_payload,
    get_mapper_options,
    get_policy_mapper_results,
)
from .simulations import (
    get_all_simulations,
    get_analysis_runs,
    get_budget_data,
    get_simulation,
)
from .store import (
    _experiment_lock,
    _experiment_queue,
    _experiments,
    _mapper_experiment_lock,
    _mapper_experiment_queue,
    _mapper_experiments,
    _save_experiments_db,
    _save_mapper_experiments_db,
)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Keep server logs compact and focused on incoming routes.
        print(f"  {self.command} {self.path}")

    def _send_json(self, data, status: int = 200):
        # All API responses use the same CORS and JSON headers.
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        # Empty POST bodies are treated as empty payloads for simpler handlers.
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode())

    def _send_html(self):
        # Serve the single-page app shell for root and frontend routes.
        body = (HERE / "index.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        # Browser preflight support for the local API.
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path

        # GET routes are read-only endpoints used to hydrate the dashboard.
        if path in ("/", "/index.html"):
            self._send_html()
        elif path == "/api/simulations":
            self._send_json(get_all_simulations())
        elif path == "/api/experiment-options":
            self._send_json(get_experiment_options())
        elif path == "/api/analysis-runs":
            self._send_json(get_analysis_runs())
        elif path == "/api/policy-mapper-results":
            self._send_json(get_policy_mapper_results())
        elif path == "/api/mapper-options":
            self._send_json(get_mapper_options())
        elif path == "/api/live-options":
            self._send_json(get_live_options())
        elif path == "/api/domain-journey-options":
            self._send_json(get_domain_journey_options())
        elif path.startswith("/api/live-sessions/"):
            # Session IDs are encoded directly after the route prefix.
            session_id = path[len("/api/live-sessions/"):]
            data, status = get_live_session(session_id)
            self._send_json(data, status)
        elif path == "/api/mapper-experiments":
            with _mapper_experiment_lock:
                data = [_compact_mapper_experiment(exp) for exp in _mapper_experiments.values()]
            self._send_json(data)
        elif path == "/api/experiments":
            with _experiment_lock:
                data = [_compact_experiment(exp) for exp in _experiments.values()]
            self._send_json(data)
        elif path == "/api/budget":
            self._send_json(get_budget_data())
        elif path.startswith("/api/simulations/"):
            # Return the raw results.json for one simulation run.
            sim_id = path[len("/api/simulations/"):]
            data = get_simulation(sim_id)
            if data:
                self._send_json(data)
            else:
                self._send_json({"error": "not found"}, 404)
        elif not path.startswith("/api/") and "." not in Path(path).name:
            # Let frontend client-side routes fall back to index.html.
            self._send_html()
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        # POST routes create work, send live-chat messages, or cancel jobs.
        if path == "/api/live-sessions":
            try:
                data = create_live_session(self._read_json())
            except Exception as exc:
                self._send_json({"error": str(exc)}, 400)
                return
            self._send_json(data, 201)
            return
        if path.startswith("/api/live-sessions/") and path.endswith("/message"):
            session_id = path[len("/api/live-sessions/"):-len("/message")]
            data, status = send_live_message(session_id, self._read_json())
            self._send_json(data, status)
            return
        if path == "/api/domain-journeys":
            try:
                data = start_domain_journey(self._read_json())
            except Exception as exc:
                self._send_json({"error": str(exc)}, 400)
                return
            self._send_json(data, 201)
            return
        if path.startswith("/api/mapper-experiments/") and path.endswith("/cancel"):
            exp_id = path[len("/api/mapper-experiments/"):-len("/cancel")]
            data, status = _cancel_mapper_experiment(exp_id)
            self._send_json(data, status)
            return
        if path == "/api/mapper-experiments":
            try:
                exp = _mapper_experiment_from_payload(self._read_json())
            except Exception as exc:
                self._send_json({"error": str(exc)}, 400)
                return
            with _mapper_experiment_lock:
                _mapper_experiments[exp.id] = exp
                _save_mapper_experiments_db()
            _ensure_mapper_worker()
            _mapper_experiment_queue.put(exp.id)
            self._send_json(_compact_mapper_experiment(exp), 201)
            return
        if path.startswith("/api/experiments/") and path.endswith("/cancel"):
            exp_id = path[len("/api/experiments/"):-len("/cancel")]
            data, status = _cancel_experiment(exp_id)
            self._send_json(data, status)
            return
        if path != "/api/experiments":
            self._send_json({"error": "not found"}, 404)
            return
        try:
            exp = _experiment_from_payload(self._read_json())
        except Exception as exc:
            self._send_json({"error": str(exc)}, 400)
            return
        with _experiment_lock:
            _experiments[exp.id] = exp
            _save_experiments_db()
        _ensure_worker()
        _experiment_queue.put(exp.id)
        self._send_json(_compact_experiment(exp), 201)
