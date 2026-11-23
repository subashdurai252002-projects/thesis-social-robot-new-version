
#!/usr/bin/env python3

"""Deterministic tests for the isolated conversation-engagement gate."""

from __future__ import annotations

import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from gaze_pipeline.runtime.conversation_engagement import (  # noqa: E402
    ConversationEngagementGate,
    EngagementConfiguration,
    EngagementEvent,
    EngagementState,
    LipMotionConfiguration,
    LipMotionEvidenceTracker,
    LipMotionState,
)


def make_mouth_scores(
    value: float,
    *,
    jaw_open: float | None = None,
) -> dict[str, float]:
    """Create one complete synthetic MediaPipe mouth measurement."""

    names = (
        "jawOpen",
        "mouthClose",
        "mouthFunnel",
        "mouthPucker",
        "mouthLeft",
        "mouthRight",
        "mouthStretchLeft",
        "mouthStretchRight",
        "mouthLowerDownLeft",
        "mouthLowerDownRight",
        "mouthUpperUpLeft",
        "mouthUpperUpRight",
    )
    scores = {name: float(value) for name in names}

    if jaw_open is not None:
        scores["jawOpen"] = float(jaw_open)

    return scores


class ConversationEngagementTests(unittest.TestCase):
    def create_engaged_gate(self) -> ConversationEngagementGate:
        gate = ConversationEngagementGate(
            owner_track_id=2,
            started_at_seconds=0.0,
        )
        result = gate.observe(
            0.0,
            owner_visible=True,
            face_available=True,
            robot_directed=True,
        )
        self.assertEqual(result.state, EngagementState.ENGAGED)
        return gate

    def test_lip_motion_requires_time_then_reports_still(self) -> None:
        tracker = LipMotionEvidenceTracker()
        evidence = None

        for index in range(5):
            evidence = tracker.observe(
                index * 0.1,
                face_available=True,
                blendshape_scores=make_mouth_scores(0.1),
            )

        self.assertIsNotNone(evidence)
        self.assertEqual(evidence.state, LipMotionState.UNCERTAIN)
        self.assertEqual(
            evidence.reason,
            "INSUFFICIENT_TEMPORAL_EVIDENCE",
        )

        for index in range(5, 10):
            evidence = tracker.observe(
                index * 0.1,
                face_available=True,
                blendshape_scores=make_mouth_scores(0.1),
            )

        self.assertEqual(
            evidence.state,
            LipMotionState.CLEARLY_STILL,
        )
        self.assertTrue(evidence.shadow_only)

    def test_lip_motion_reports_strong_repeated_motion(self) -> None:
        tracker = LipMotionEvidenceTracker()
        evidence = None

        for index in range(10):
            value = 0.05 if index % 2 == 0 else 0.45
            evidence = tracker.observe(
                index * 0.1,
                face_available=True,
                blendshape_scores=make_mouth_scores(value),
            )

        self.assertEqual(
            evidence.state,
            LipMotionState.SUPPORTS_SPEECH,
        )
        self.assertEqual(
            evidence.reason,
            "PROVISIONAL_MOTION_SUPPORT",
        )

    def test_lip_motion_keeps_moderate_motion_uncertain(self) -> None:
        tracker = LipMotionEvidenceTracker()
        evidence = None

        for index in range(10):
            value = 0.100 if index % 2 == 0 else 0.115
            evidence = tracker.observe(
                index * 0.1,
                face_available=True,
                blendshape_scores=make_mouth_scores(value),
            )

        self.assertEqual(evidence.state, LipMotionState.UNCERTAIN)
        self.assertEqual(
            evidence.reason,
            "AMBIGUOUS_MOUTH_MOTION",
        )

    def test_lip_motion_missing_face_is_uncertain(self) -> None:
        tracker = LipMotionEvidenceTracker()
        evidence = None

        for index in range(11):
            evidence = tracker.observe(
                index * 0.1,
                face_available=False,
            )

        self.assertEqual(evidence.state, LipMotionState.UNCERTAIN)
        self.assertEqual(evidence.face_coverage, 0.0)

    def test_lip_motion_validation_and_immutability(self) -> None:
        with self.assertRaises(ValueError):
            LipMotionConfiguration(
                clearly_still_max_score=0.02,
                supports_speech_min_score=0.01,
            )

        tracker = LipMotionEvidenceTracker()
        evidence = tracker.observe(
            1.0,
            face_available=True,
            blendshape_scores=make_mouth_scores(0.1),
        )

        with self.assertRaises(ValueError):
            tracker.observe(
                0.5,
                face_available=True,
                blendshape_scores=make_mouth_scores(0.1),
            )

        with self.assertRaises(ValueError):
            LipMotionEvidenceTracker().observe(
                0.0,
                face_available=True,
                blendshape_scores={"jawOpen": 0.1},
            )

        with self.assertRaises(ValueError):
            LipMotionEvidenceTracker().observe(
                0.0,
                face_available=False,
                blendshape_scores=make_mouth_scores(0.1),
            )

        with self.assertRaises(FrozenInstanceError):
            evidence.state = LipMotionState.SUPPORTS_SPEECH

    def test_engaged_owner_speech_is_accepted(self) -> None:
        gate = self.create_engaged_gate()

        decision = gate.evaluate_speech(
            0.5,
            speaker_matches_owner=True,
        )

        self.assertTrue(decision.accepted)
        self.assertEqual(
            decision.event,
            EngagementEvent.SPEECH_ACCEPTED_SPEAKER_VERIFIED,
        )

    def test_unverified_speaker_is_reported(self) -> None:
        gate = self.create_engaged_gate()

        decision = gate.evaluate_speech(0.5)

        self.assertTrue(decision.accepted)
        self.assertEqual(
            decision.event,
            EngagementEvent.SPEECH_ACCEPTED_SPEAKER_UNVERIFIED,
        )

    def test_speaker_mismatch_is_rejected(self) -> None:
        gate = self.create_engaged_gate()

        decision = gate.evaluate_speech(
            0.5,
            speaker_matches_owner=False,
        )

        self.assertFalse(decision.accepted)
        self.assertEqual(
            decision.event,
            EngagementEvent.SPEECH_REJECTED_SPEAKER_MISMATCH,
        )

    def test_brief_look_away_is_allowed(self) -> None:
        gate = self.create_engaged_gate()

        update = gate.observe(
            1.0,
            owner_visible=True,
            face_available=True,
            robot_directed=False,
        )
        decision = gate.evaluate_speech(1.1)

        self.assertEqual(
            update.state,
            EngagementState.TEMPORARILY_UNCERTAIN,
        )
        self.assertTrue(update.retain_owner)
        self.assertTrue(decision.accepted)

    def test_sustained_look_away_rejects_speech(self) -> None:
        gate = self.create_engaged_gate()

        gate.observe(
            1.0,
            owner_visible=True,
            face_available=True,
            robot_directed=False,
        )
        update = gate.observe(
            3.1,
            owner_visible=True,
            face_available=True,
            robot_directed=False,
        )
        decision = gate.evaluate_speech(3.2)

        self.assertEqual(update.state, EngagementState.SUSPENDED)
        self.assertFalse(decision.accepted)
        self.assertEqual(
            decision.event,
            EngagementEvent.SPEECH_REJECTED_SUSTAINED_AWAY,
        )

    def test_covered_face_rejects_speech(self) -> None:
        gate = self.create_engaged_gate()

        update = gate.observe(
            1.0,
            owner_visible=True,
            face_available=False,
            robot_directed=False,
        )
        decision = gate.evaluate_speech(1.1)

        self.assertTrue(update.retain_owner)
        self.assertFalse(decision.accepted)
        self.assertEqual(
            decision.event,
            EngagementEvent.SPEECH_REJECTED_FACE_MISSING,
        )

    def test_stale_visual_evidence_rejects_speech(self) -> None:
        gate = self.create_engaged_gate()

        decision = gate.evaluate_speech(2.1)

        self.assertFalse(decision.accepted)
        self.assertEqual(
            decision.event,
            EngagementEvent.SPEECH_REJECTED_STALE_VISUAL_EVIDENCE,
        )

    def test_owner_can_reengage(self) -> None:
        gate = self.create_engaged_gate()

        gate.observe(
            1.0,
            owner_visible=True,
            face_available=True,
            robot_directed=False,
        )
        gate.observe(
            3.1,
            owner_visible=True,
            face_available=True,
            robot_directed=False,
        )
        update = gate.observe(
            3.5,
            owner_visible=True,
            face_available=True,
            robot_directed=True,
        )
        decision = gate.evaluate_speech(3.6)

        self.assertEqual(update.state, EngagementState.ENGAGED)
        self.assertEqual(
            update.event,
            EngagementEvent.OWNER_REENGAGED,
        )
        self.assertTrue(decision.accepted)

    def test_waiting_person_cannot_steal_active_owner(self) -> None:
        gate = self.create_engaged_gate()

        update = gate.observe(
            1.0,
            owner_visible=True,
            face_available=True,
            robot_directed=True,
            waiting_candidate_ready=True,
        )

        self.assertEqual(update.state, EngagementState.ENGAGED)
        self.assertFalse(update.allow_handover)
        self.assertTrue(update.retain_owner)

    def test_missing_owner_allows_handover_after_two_seconds(self) -> None:
        gate = self.create_engaged_gate()

        gate.observe(
            1.0,
            owner_visible=False,
            face_available=False,
            robot_directed=False,
            waiting_candidate_ready=True,
        )
        update = gate.observe(
            3.0,
            owner_visible=False,
            face_available=False,
            robot_directed=False,
            waiting_candidate_ready=True,
        )

        self.assertEqual(update.state, EngagementState.RELEASED)
        self.assertTrue(update.release_owner)
        self.assertTrue(update.allow_handover)
        self.assertEqual(
            update.event,
            EngagementEvent.OWNER_RELEASED_FOR_HANDOVER,
        )

    def test_missing_owner_releases_after_five_seconds(self) -> None:
        gate = self.create_engaged_gate()

        gate.observe(
            1.0,
            owner_visible=False,
            face_available=False,
            robot_directed=False,
        )
        update = gate.observe(
            6.0,
            owner_visible=False,
            face_available=False,
            robot_directed=False,
        )

        self.assertEqual(update.state, EngagementState.RELEASED)
        self.assertEqual(
            update.event,
            EngagementEvent.OWNER_RELEASED_ABSENCE_TIMEOUT,
        )

    def test_owner_return_before_timeout_keeps_conversation(self) -> None:
        gate = self.create_engaged_gate()

        gate.observe(
            1.0,
            owner_visible=False,
            face_available=False,
            robot_directed=False,
        )
        update = gate.observe(
            1.8,
            owner_visible=True,
            face_available=True,
            robot_directed=True,
        )

        self.assertEqual(update.state, EngagementState.ENGAGED)
        self.assertTrue(update.retain_owner)
        self.assertFalse(update.release_owner)

    def test_released_owner_speech_is_rejected(self) -> None:
        gate = self.create_engaged_gate()

        gate.observe(
            1.0,
            owner_visible=False,
            face_available=False,
            robot_directed=False,
        )
        gate.observe(
            6.0,
            owner_visible=False,
            face_available=False,
            robot_directed=False,
        )
        decision = gate.evaluate_speech(6.1)

        self.assertFalse(decision.accepted)
        self.assertFalse(decision.retain_owner)
        self.assertEqual(
            decision.event,
            EngagementEvent.SPEECH_REJECTED_OWNER_RELEASED,
        )

    def test_results_are_immutable(self) -> None:
        gate = self.create_engaged_gate()
        snapshot = gate.get_snapshot()

        with self.assertRaises(FrozenInstanceError):
            snapshot.owner_visible = False

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ConversationEngagementGate(
                owner_track_id=-1,
                started_at_seconds=0.0,
            )

        with self.assertRaises(ValueError):
            EngagementConfiguration(
                waiting_candidate_handover_seconds=6.0,
                owner_absence_timeout_seconds=5.0,
            )

        gate = self.create_engaged_gate()

        with self.assertRaises(ValueError):
            gate.observe(
                1.0,
                owner_visible=False,
                face_available=True,
                robot_directed=False,
            )

        with self.assertRaises(ValueError):
            gate.observe(
                1.0,
                owner_visible=True,
                face_available=False,
                robot_directed=True,
            )

        with self.assertRaises(ValueError):
            gate.evaluate_speech(
                1.0,
                speaker_matches_owner="yes",
            )

        with self.assertRaises(ValueError):
            gate.evaluate_speech(-1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
