
#!/usr/bin/env python3

"""Read-only local dashboard state and HTTP server."""

from __future__ import annotations

import copy
import json
import math
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 8000
ASSET_DIRECTORY = Path(__file__).resolve().parents[1] / "dashboard"
MAX_FRAME_BYTES = 4 * 1024 * 1024

DEFAULT_STATE = {
    "runtime": {
        "phase": "STARTING",
        "message": "Preparing robot resources",
        "interaction_active": False,
    },
    "perception": {
        "detected_people": 0,
        "owner_track_id": None,
        "attention": "WAITING",
        "liveness": "WAITING",
        "engagement": "WAITING",
        "face_available": None,
    },
    "identity": {
        "status": "NOT_CHECKED",
        "recognition_ms": None,
    },
    "conversation": {
        "transcript": "",
        "response": "",
        "route": "NONE",
        "speech_profile": "normal",
        "completed_turns": 0,
    },
    "metrics": {
        "whisper_seconds": None,
        "speech_end_to_text_seconds": None,
        "openai_seconds": None,
        "piper_seconds": None,
        "response_start_seconds": None,
        "playback_seconds": None,
    },
    "modules": {
        "camera": "STARTING",
        "person_detection": "STARTING",
        "face_detection": "STARTING",
        "gaze_attention": "STARTING",
        "liveness": "STARTING",
        "whisper": "STARTING",
        "openai": "READY_WHEN_REQUESTED",
        "piper": "STARTING",
        "returning_user": "STARTING",
        "respeaker_doa": "LAB_VALIDATION_PENDING",
        "speaker_attribution": "LAB_VALIDATION_PENDING",
        "jetson_ros2_unitree": "NOT_INTEGRATED",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DashboardState:
    """Thread-safe, in-memory, sanitized dashboard state."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._started = time.monotonic()
        self._updated_at = utc_now()
        self._sequence = 0
        self._state = copy.deepcopy(DEFAULT_STATE)
        self._events: deque[dict[str, object]] = deque(maxlen=40)
        self._frame_jpeg: bytes | None = None
        self._visitor_frame_jpeg: bytes | None = None

    @staticmethod
    def _safe_value(value: object) -> object:
        if value is None or isinstance(value, bool):
            return value

        if isinstance(value, int):
            return value

        if isinstance(value, float):
            if not math.isfinite(value):
                return None
            return round(value, 3)

        text = str(value).strip()
        return text[:1500]

    def update(self, section: str, **values: object) -> None:
        with self._lock:
            if section not in self._state:
                raise ValueError(f"Unknown dashboard section: {section}")

            target = self._state[section]

            for key, value in values.items():
                if key not in target:
                    raise ValueError(
                        f"Unknown dashboard field: {section}.{key}"
                    )
                target[key] = self._safe_value(value)

            self._sequence += 1
            self._updated_at = utc_now()

    def record_event(self, category: str, message: str) -> None:
        safe_category = str(category).strip()[:40] or "system"
        safe_message = str(message).strip()[:300]

        if not safe_message:
            return

        with self._lock:
            self._events.appendleft(
                {
                    "time": utc_now(),
                    "category": safe_category,
                    "message": safe_message,
                }
            )
            self._sequence += 1
            self._updated_at = utc_now()

    def reset_conversation(self) -> None:
        with self._lock:
            self._state["conversation"] = copy.deepcopy(
                DEFAULT_STATE["conversation"]
            )
            self._state["metrics"] = copy.deepcopy(
                DEFAULT_STATE["metrics"]
            )
            self._sequence += 1
            self._updated_at = utc_now()

    @staticmethod
    def _validated_frame_jpeg(frame_jpeg: bytes) -> bytes:
        if not isinstance(frame_jpeg, bytes):
            raise TypeError("frame_jpeg must be bytes.")

        if not frame_jpeg or len(frame_jpeg) > MAX_FRAME_BYTES:
            raise ValueError("frame_jpeg has an invalid size.")

        if not frame_jpeg.startswith(b"\xff\xd8"):
            raise ValueError("frame_jpeg must contain JPEG data.")

        return frame_jpeg

    def update_frame_jpeg(self, frame_jpeg: bytes) -> None:
        """Replace the latest ephemeral annotated camera frame."""

        validated = self._validated_frame_jpeg(frame_jpeg)

        with self._lock:
            self._frame_jpeg = validated

    def update_visitor_frame_jpeg(self, frame_jpeg: bytes) -> None:
        """Replace the latest ephemeral clean visitor camera frame."""

        validated = self._validated_frame_jpeg(frame_jpeg)

        with self._lock:
            self._visitor_frame_jpeg = validated

    def get_frame_jpeg(self) -> bytes | None:
        """Return the latest frame without persisting it."""

        with self._lock:
            return self._frame_jpeg

    def get_visitor_frame_jpeg(self) -> bytes | None:
        """Return the latest clean visitor frame without persisting it."""

        with self._lock:
            return self._visitor_frame_jpeg

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            result = copy.deepcopy(self._state)
            result.update(
                {
                    "schema_version": 1,
                    "sequence": self._sequence,
                    "updated_at": self._updated_at,
                    "uptime_seconds": round(
                        time.monotonic() - self._started,
                        1,
                    ),
                    "events": list(self._events),
                    "privacy": {
                        "local_only": True,
                        "read_only": True,
                        "media_streaming": True,
                        "persistent_dashboard_storage": False,
                    },
                }
            )
            return result


class DashboardHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def create_request_handler(state: DashboardState):
    """Create one request handler bound to the supplied state."""

    routes = {
        "/": ("visitor.html", "text/html; charset=utf-8"),
        "/visitor": ("visitor.html", "text/html; charset=utf-8"),
        "/research": ("research.html", "text/html; charset=utf-8"),
        "/styles.css": ("styles.css", "text/css; charset=utf-8"),
        "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    }

    class DashboardRequestHandler(BaseHTTPRequestHandler):
        server_version = "SocialRobotDashboard/1.0"

        def log_message(self, _format: str, *_args: object) -> None:
            return

        def _send(
            self,
            status: int,
            body: bytes,
            content_type: str,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                (
                    "default-src 'self'; "
                    "script-src 'self'; "
                    "style-src 'self'; "
                    "img-src 'self' data:; "
                    "connect-src 'self'; "
                    "frame-ancestors 'none'; "
                    "base-uri 'none'"
                ),
            )
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlsplit(self.path).path

            if path == "/api/state":
                payload = json.dumps(
                    state.snapshot(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                self._send(
                    200,
                    payload,
                    "application/json; charset=utf-8",
                )
                return

            if path == "/api/frame.jpg":
                frame_jpeg = state.get_frame_jpeg()

                if frame_jpeg is None:
                    self._send(
                        404,
                        b"Frame not available",
                        "text/plain; charset=utf-8",
                    )
                    return

                self._send(200, frame_jpeg, "image/jpeg")
                return

            if path == "/api/visitor-frame.jpg":
                frame_jpeg = state.get_visitor_frame_jpeg()

                if frame_jpeg is None:
                    self._send(
                        404,
                        b"Frame not available",
                        "text/plain; charset=utf-8",
                    )
                    return

                self._send(200, frame_jpeg, "image/jpeg")
                return

            if path == "/health":
                self._send(
                    200,
                    b'{"status":"ok"}',
                    "application/json; charset=utf-8",
                )
                return

            route = routes.get(path)

            if route is None:
                self._send(
                    404,
                    b"Not found",
                    "text/plain; charset=utf-8",
                )
                return

            filename, content_type = route
            asset_path = ASSET_DIRECTORY / filename

            try:
                body = asset_path.read_bytes()
            except OSError:
                self._send(
                    503,
                    b"Dashboard asset unavailable",
                    "text/plain; charset=utf-8",
                )
                return

            self._send(200, body, content_type)

        def do_POST(self) -> None:
            self._send(
                405,
                b"Read-only dashboard",
                "text/plain; charset=utf-8",
            )

    return DashboardRequestHandler


class LocalDashboardServer:
    """Run the read-only dashboard in one isolated daemon thread."""

    def __init__(
        self,
        state: DashboardState,
        host: str = DASHBOARD_HOST,
        port: int = DASHBOARD_PORT,
    ) -> None:
        if host != DASHBOARD_HOST:
            raise ValueError(
                "Dashboard must remain bound to 127.0.0.1."
            )

        self.state = state
        self.host = host
        self.port = int(port)
        self._server: DashboardHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> str:
        if self._server is not None:
            return self.url

        handler = create_request_handler(self.state)
        self._server = DashboardHTTPServer(
            (self.host, self.port),
            handler,
        )
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="local-dashboard",
            daemon=True,
        )
        self._thread.start()
        return self.url

    def stop(self) -> None:
        server = self._server
        thread = self._thread
        self._server = None
        self._thread = None

        if server is None:
            return

        server.shutdown()
        server.server_close()

        if thread is not None:
            thread.join(timeout=2.0)


__all__ = [
    "DASHBOARD_HOST",
    "DASHBOARD_PORT",
    "DashboardState",
    "LocalDashboardServer",
]
