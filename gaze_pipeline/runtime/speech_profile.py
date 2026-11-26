"""Deterministic local speech-profile intent handling.

This module contains decision logic only. It does not synthesize speech,
play audio, call OpenAI, use the microphone, or save persistent preferences.
"""

from __future__ import annotations

from dataclasses import dataclass

from live_streaming_stt_local_agreement import normalize_words


@dataclass(frozen=True)
class SpeechProfile:
    """One validated Piper speech-speed profile."""

    name: str
    length_scale: float


@dataclass(frozen=True)
class SpeechProfileIntent:
    """Structured result of one local speech-profile request."""

    matched: bool
    profile_name: str | None
    control_only: bool
    remaining_text: str
    ambiguous: bool
    reason: str
    confirmation_text: str


SPEECH_PROFILES = {
    "fast": SpeechProfile(
        name="fast",
        length_scale=0.70,
    ),
    "normal": SpeechProfile(
        name="normal",
        length_scale=1.00,
    ),
    "slow": SpeechProfile(
        name="slow",
        length_scale=1.40,
    ),
}


POLITE_PREFIXES = (
    ("so",),
    ("okay",),
    ("ok",),
    ("well",),
    ("i", "asked"),
    ("i", "said"),
    ("could", "you"),
    ("would", "you"),
    ("will", "you"),
    ("can", "you"),
    ("please",),
)


REQUEST_PATTERNS = (
    # Slow requests
    (
        "slow",
        "slow",
        ("speak", "a", "little", "slower"),
    ),
    (
        "slow",
        "slow",
        ("speak", "a", "bit", "slower"),
    ),
    (
        "slow",
        "slow",
        ("speak", "more", "slowly"),
    ),
    (
        "slow",
        "slow",
        ("speak", "slower"),
    ),
    (
        "slow",
        "slow",
        ("speak", "slowly"),
    ),
    (
        "slow",
        "slow",
        ("talk", "a", "little", "slower"),
    ),
    (
        "slow",
        "slow",
        ("talk", "a", "bit", "slower"),
    ),
    (
        "slow",
        "slow",
        ("talk", "more", "slowly"),
    ),
    (
        "slow",
        "slow",
        ("talk", "slower"),
    ),
    (
        "slow",
        "slow",
        ("talk", "slowly"),
    ),
    (
        "slow",
        "slow",
        ("slow", "down"),
    ),
    (
        "slow",
        "slow",
        ("you", "are", "speaking", "too", "fast"),
    ),
    (
        "slow",
        "slow",
        ("you", "are", "talking", "too", "fast"),
    ),
    (
        "slow",
        "slow",
        ("you", "re", "speaking", "too", "fast"),
    ),
    (
        "slow",
        "slow",
        ("you", "re", "talking", "too", "fast"),
    ),
    (
        "slow",
        "slow",
        ("you're", "speaking", "too", "fast"),
    ),
    (
        "slow",
        "slow",
        ("you're", "talking", "too", "fast"),
    ),

    # Fast requests
    (
        "fast",
        "fast",
        ("speak", "a", "little", "faster"),
    ),
    (
        "fast",
        "fast",
        ("speak", "a", "bit", "faster"),
    ),
    (
        "fast",
        "fast",
        ("speak", "more", "quickly"),
    ),
    (
        "fast",
        "fast",
        ("speak", "fastly"),
    ),
    (
        "fast",
        "fast",
        ("speak", "fast"),
    ),
    (
        "fast",
        "fast",
        ("speak", "faster"),
    ),
    (
        "fast",
        "fast",
        ("talk", "a", "little", "faster"),
    ),
    (
        "fast",
        "fast",
        ("talk", "a", "bit", "faster"),
    ),
    (
        "fast",
        "fast",
        ("talk", "more", "quickly"),
    ),
    (
        "fast",
        "fast",
        ("talk", "fastly"),
    ),
    (
        "fast",
        "fast",
        ("talk", "fast"),
    ),
    (
        "fast",
        "fast",
        ("talk", "faster"),
    ),
    (
        "fast",
        "fast",
        ("speed", "up"),
    ),
    (
        "fast",
        "fast",
        ("go", "fastly"),
    ),
    (
        "fast",
        "fast",
        ("you", "are", "speaking", "too", "slowly"),
    ),
    (
        "fast",
        "fast",
        ("you", "are", "talking", "too", "slowly"),
    ),
    (
        "fast",
        "fast",
        ("you", "re", "speaking", "too", "slowly"),
    ),
    (
        "fast",
        "fast",
        ("you", "re", "talking", "too", "slowly"),
    ),
    (
        "fast",
        "fast",
        ("you're", "speaking", "too", "slowly"),
    ),
    (
        "fast",
        "fast",
        ("you're", "talking", "too", "slowly"),
    ),

    # Normal/reset requests
    (
        "normal",
        "normal",
        ("go", "back", "to", "normal", "speed"),
    ),
    (
        "normal",
        "normal",
        ("speak", "at", "normal", "speed"),
    ),
    (
        "normal",
        "normal",
        ("speak", "in", "normal", "speed"),
    ),
    (
        "normal",
        "normal",
        ("speak", "at", "medium", "speed"),
    ),
    (
        "normal",
        "normal",
        ("speak", "in", "medium", "speed"),
    ),
    (
        "normal",
        "normal",
        ("speak", "at", "a", "medium", "pace"),
    ),
    (
        "normal",
        "normal",
        ("speak", "in", "a", "medium", "pace"),
    ),
    (
        "normal",
        "normal",
        ("speak", "normal"),
    ),
    (
        "normal",
        "normal",
        ("speak", "normally"),
    ),
    (
        "normal",
        "normal",
        ("talk", "at", "normal", "speed"),
    ),
    (
        "normal",
        "normal",
        ("talk", "in", "normal", "speed"),
    ),
    (
        "normal",
        "normal",
        ("talk", "at", "medium", "speed"),
    ),
    (
        "normal",
        "normal",
        ("talk", "in", "medium", "speed"),
    ),
    (
        "normal",
        "normal",
        ("talk", "normal"),
    ),
    (
        "normal",
        "normal",
        ("talk", "normally"),
    ),
    (
        "normal",
        "normal",
        ("use", "your", "usual", "speed"),
    ),
    (
        "normal",
        "normal",
        ("use", "normal", "speed"),
    ),
    (
        "normal",
        "normal",
        ("use", "medium", "speed"),
    ),
    (
        "normal",
        "normal",
        ("reset", "your", "speaking", "speed"),
    ),
    (
        "normal",
        "normal",
        ("reset", "your", "speed"),
    ),
    (
        "normal",
        "normal",
        ("normal", "speed"),
    ),
    (
        "normal",
        "normal",
        ("medium", "speed"),
    ),
    (
        "normal",
        "normal",
        ("medium", "pace"),
    ),

    # Clarity requests use the tested slow profile.
    (
        "slow",
        "clarity",
        ("speak", "more", "clearly"),
    ),
    (
        "slow",
        "clarity",
        ("speak", "clearly"),
    ),
    (
        "slow",
        "clarity",
        ("talk", "more", "clearly"),
    ),
    (
        "slow",
        "clarity",
        ("talk", "clearly"),
    ),
)


CONFIRMATIONS = {
    "fast": "Okay. I'll speak faster.",
    "normal": "Okay. I'll use my normal speed.",
    "slow": "Okay. I'll speak more slowly.",
    "clarity": "Okay. I'll speak more slowly and clearly.",
}


NEGATION_FRAGMENTS = {
    "do not",
    "don't",
    "dont",
    "don t",
    "never",
}


CONNECTOR_WORDS = {
    "and",
    "also",
    "then",
}


CONFLICT_CUES = {
    "slow": {
        ("slow",),
        ("slowly",),
        ("slower",),
        ("more", "slowly"),
    },
    "fast": {
        ("fast",),
        ("fastly",),
        ("faster",),
        ("quickly",),
        ("more", "quickly"),
    },
    "normal": {
        ("medium",),
        ("medium", "pace"),
        ("medium", "speed"),
        ("normal",),
        ("normal", "speed"),
        ("usual", "speed"),
    },
}


def get_speech_profile(profile_name: str) -> SpeechProfile:
    """Return one validated profile by name."""

    if not isinstance(profile_name, str):
        raise TypeError("profile_name must be a string.")

    try:
        return SPEECH_PROFILES[profile_name]
    except KeyError as error:
        raise ValueError(
            f"Unknown speech profile: {profile_name!r}."
        ) from error


def piper_length_scale_arguments(
    profile_name: str,
) -> tuple[str, str]:
    """Return Piper CLI arguments for one validated profile."""

    profile = get_speech_profile(profile_name)

    return (
        "--length-scale",
        f"{profile.length_scale:.2f}",
    )


def _strip_polite_prefixes(
    words: list[str],
) -> list[str]:
    """Remove repeated leading polite request words."""

    remaining = list(words)

    while remaining:
        changed = False

        for prefix in POLITE_PREFIXES:
            prefix_length = len(prefix)

            if tuple(remaining[:prefix_length]) == prefix:
                remaining = remaining[prefix_length:]
                changed = True
                break

        if not changed:
            break

    return remaining


def _starts_with(
    words: list[str],
    pattern: tuple[str, ...],
) -> bool:
    return tuple(words[:len(pattern)]) == pattern


def _strip_combined_request_connector(
    words: list[str],
) -> list[str]:
    """Remove only the connector after a matched control clause."""

    remaining = list(words)

    while (
        remaining
        and remaining[0] in CONNECTOR_WORDS
    ):
        remaining.pop(0)

    if remaining == ["please"]:
        return []

    return remaining


def _has_conflicting_following_request(
    words: list[str],
    selected_profile: str,
) -> bool:
    """Detect a second incompatible speed request after the first."""

    remaining = list(words)

    while (
        remaining
        and remaining[0] in CONNECTOR_WORDS
    ):
        remaining.pop(0)

    if (
        remaining
        and remaining[0] in {"speak", "talk"}
    ):
        remaining.pop(0)

    for profile_name, cues in CONFLICT_CUES.items():
        if profile_name == selected_profile:
            continue

        if any(
            _starts_with(remaining, cue)
            for cue in cues
        ):
            return True

    return False


def parse_speech_profile_intent(
    text: str,
) -> SpeechProfileIntent:
    """Parse one robot-directed local speech-profile request."""

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    words = normalize_words(text)

    if not words:
        return SpeechProfileIntent(
            matched=False,
            profile_name=None,
            control_only=False,
            remaining_text="",
            ambiguous=False,
            reason="empty_or_nonmeaningful_text",
            confirmation_text="",
        )

    normalized_text = " ".join(words)

    if any(
        fragment in normalized_text
        for fragment in NEGATION_FRAGMENTS
    ):
        return SpeechProfileIntent(
            matched=False,
            profile_name=None,
            control_only=False,
            remaining_text=normalized_text,
            ambiguous=False,
            reason="negated_speed_request",
            confirmation_text="",
        )

    request_words = _strip_polite_prefixes(words)

    if (
        request_words
        and request_words[0] in {"speak", "talk", "make", "think"}
    ):
        requested_directions = set()
        request_word_set = set(request_words)

        if request_word_set & {
            "slow",
            "slowly",
            "slower",
        }:
            requested_directions.add("slow")

        if request_word_set & {
            "fast",
            "fastly",
            "faster",
            "quickly",
        }:
            requested_directions.add("fast")

        if request_word_set & {
            "medium",
            "normal",
        }:
            requested_directions.add("normal")

        if len(requested_directions) > 1:
            return SpeechProfileIntent(
                matched=True,
                profile_name=None,
                control_only=False,
                remaining_text="",
                ambiguous=True,
                reason="conflicting_speed_requests",
                confirmation_text="",
            )

    for (
        profile_name,
        confirmation_kind,
        pattern,
    ) in REQUEST_PATTERNS:
        if not _starts_with(request_words, pattern):
            continue

        if (
            pattern == ("normal", "speed")
            and len(request_words) != len(pattern)
        ):
            continue

        unconsumed_words = request_words[len(pattern):]

        if _has_conflicting_following_request(
            unconsumed_words,
            profile_name,
        ):
            return SpeechProfileIntent(
                matched=True,
                profile_name=None,
                control_only=False,
                remaining_text="",
                ambiguous=True,
                reason="conflicting_speed_requests",
                confirmation_text="",
            )

        remaining_words = (
            _strip_combined_request_connector(
                unconsumed_words
            )
        )
        remaining_text = " ".join(remaining_words)

        return SpeechProfileIntent(
            matched=True,
            profile_name=profile_name,
            control_only=not bool(remaining_text),
            remaining_text=remaining_text,
            ambiguous=False,
            reason=(
                f"{confirmation_kind}_profile_request"
            ),
            confirmation_text=CONFIRMATIONS[
                confirmation_kind
            ],
        )

    return SpeechProfileIntent(
        matched=False,
        profile_name=None,
        control_only=False,
        remaining_text=normalized_text,
        ambiguous=False,
        reason="no_speech_profile_request",
        confirmation_text="",
    )


def combined_request_needs_repetition(text: str) -> bool:
    """Return whether a preserved combined request appears truncated."""

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    words = normalize_words(text)

    if not words:
        return True

    if words[-1] in {
        "a",
        "about",
        "an",
        "and",
        "for",
        "of",
        "or",
        "so",
        "the",
        "to",
        "with",
    }:
        return True

    return len(words) == 1


def incomplete_speed_fragment_needs_repetition(text: str) -> bool:
    """Detect a truncated speed-control suffix without guessing its intent."""

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    words = normalize_words(text)

    if not words:
        return False

    request_words = _strip_polite_prefixes(words)

    if tuple(request_words) in {
        ("speak", "first"),
        ("talk", "first"),
    }:
        return True

    if words[0] not in {"and", "or"}:
        return False

    fragment_words = [
        word
        for word in words[1:]
        if word != "please"
    ]
    speed_words = {
        "fast",
        "faster",
        "normal",
        "quickly",
        "slow",
        "slower",
        "slowly",
        "speak",
        "speed",
        "talk",
    }
    speed_cues = {
        "fast",
        "faster",
        "normal",
        "quickly",
        "slow",
        "slower",
        "slowly",
    }

    return (
        bool(set(fragment_words) & speed_cues)
        and set(fragment_words) <= speed_words
    )


def parse_speech_profile_clarification(
    text: str,
) -> str | None:
    """Return a profile selected after an ambiguous-speed question."""

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    words = normalize_words(text)

    if not words:
        return None

    normalized_text = " ".join(words)

    if any(
        fragment in normalized_text
        for fragment in NEGATION_FRAGMENTS
    ):
        return None

    answer_words = _strip_polite_prefixes(words)

    while (
        answer_words
        and answer_words[-1] == "please"
    ):
        answer_words.pop()

    accepted_answers = {
        "slow": {
            ("slow",),
            ("slower",),
            ("more", "slowly"),
            ("a", "little", "slower"),
            ("a", "bit", "slower"),
            ("speak", "slowly"),
            ("speak", "slower"),
            ("talk", "slowly"),
            ("talk", "slower"),
            ("more", "clearly"),
            ("speak", "clearly"),
            ("speak", "more", "clearly"),
        },
        "fast": {
            ("fast",),
            ("faster",),
            ("more", "quickly"),
            ("a", "little", "faster"),
            ("a", "bit", "faster"),
            ("speak", "faster"),
            ("talk", "faster"),
            ("speed", "up"),
        },
        "normal": {
            ("normal",),
            ("normal", "speed"),
            ("usual", "speed"),
            ("use", "normal", "speed"),
            ("use", "your", "usual", "speed"),
            ("speak", "at", "normal", "speed"),
            ("speak", "in", "normal", "speed"),
            ("talk", "in", "normal", "speed"),
            ("go", "back", "to", "normal", "speed"),
        },
    }

    normalized_answer = tuple(answer_words)

    matches = [
        profile_name
        for profile_name, answers in accepted_answers.items()
        if normalized_answer in answers
    ]

    if len(matches) != 1:
        return None

    return matches[0]
