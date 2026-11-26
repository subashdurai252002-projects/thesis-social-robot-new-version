
#!/usr/bin/env python3

"""Tests for the local read-only dashboard."""

import json
import unittest
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from gaze_pipeline.runtime.dashboard import (
    DashboardState,
    LocalDashboardServer,
)


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.opener = build_opener(ProxyHandler({}))

    def test_local_read_only_state(self):
        state = DashboardState().snapshot()

        self.assertTrue(state["privacy"]["local_only"])
        self.assertTrue(state["privacy"]["read_only"])
        self.assertTrue(state["privacy"]["media_streaming"])
        self.assertFalse(
            state["privacy"]["persistent_dashboard_storage"]
        )
        self.assertNotIn("user_id", json.dumps(state))
        self.assertNotIn("embedding", json.dumps(state))

    def test_state_update(self):
        state = DashboardState()
        state.update(
            "runtime",
            phase="LISTENING",
            message="Waiting for speech",
            interaction_active=True,
        )
        state.update(
            "conversation",
            transcript="Hello robot",
            speech_profile="fast",
        )
        state.record_event("speech", "Accepted transcription")

        result = state.snapshot()

        self.assertEqual(result["runtime"]["phase"], "LISTENING")
        self.assertEqual(
            result["conversation"]["transcript"],
            "Hello robot",
        )
        self.assertEqual(
            result["events"][0]["message"],
            "Accepted transcription",
        )

    def test_unknown_fields_rejected(self):
        state = DashboardState()

        with self.assertRaises(ValueError):
            state.update("private_profiles", count=1)

        with self.assertRaises(ValueError):
            state.update("runtime", secret="value")

    def test_local_server(self):
        state = DashboardState()
        frame_jpeg = b"\xff\xd8dashboard-test-frame"
        visitor_frame_jpeg = b"\xff\xd8visitor-test-frame"
        state.update_frame_jpeg(frame_jpeg)
        state.update_visitor_frame_jpeg(visitor_frame_jpeg)
        server = LocalDashboardServer(state, port=0)

        try:
            url = server.start()

            for path in (
                "/visitor",
                "/research",
                "/styles.css",
                "/app.js",
            ):
                response = self.opener.open(url + path, timeout=2)
                self.assertEqual(response.status, 200)
                self.assertEqual(
                    response.headers["Cache-Control"],
                    "no-store",
                )

            response = self.opener.open(
                url + "/api/state",
                timeout=2,
            )
            payload = json.loads(response.read().decode("utf-8"))
            self.assertEqual(payload["schema_version"], 1)

            health = self.opener.open(url + "/health", timeout=2)
            self.assertEqual(
                json.loads(health.read().decode("utf-8")),
                {"status": "ok"},
            )

            frame = self.opener.open(
                url + "/api/frame.jpg",
                timeout=2,
            )
            self.assertEqual(frame.headers["Content-Type"], "image/jpeg")
            self.assertEqual(frame.read(), frame_jpeg)

            visitor_frame = self.opener.open(
                url + "/api/visitor-frame.jpg",
                timeout=2,
            )
            self.assertEqual(
                visitor_frame.headers["Content-Type"],
                "image/jpeg",
            )
            self.assertEqual(visitor_frame.read(), visitor_frame_jpeg)
        finally:
            server.stop()

    def test_server_rejects_writes(self):
        server = LocalDashboardServer(DashboardState(), port=0)

        try:
            url = server.start()
            request = Request(
                url + "/api/state",
                data=b"{}",
                method="POST",
            )

            with self.assertRaises(HTTPError) as error:
                self.opener.open(request, timeout=2)

            self.assertEqual(error.exception.code, 405)
        finally:
            server.stop()

    def test_non_local_bind_rejected(self):
        with self.assertRaises(ValueError):
            LocalDashboardServer(
                DashboardState(),
                host="0.0.0.0",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
