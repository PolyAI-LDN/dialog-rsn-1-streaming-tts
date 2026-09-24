"""What may reach the synthesizer: emotion tags off, unspeakable text held back."""

from __future__ import annotations

import re

TAG_RE = re.compile(r"^\[(\w+)\]\s*")
MAX_TAG_LENGTH = 24   # hold at most this many characters while they might still be a tag
UNSPEAKABLE = " \t\r\n.,!?;:-"


def speakable(text: str) -> bool:
    """False for empty and punctuation-only text, which most TTS vendors reject."""
    return text.strip(UNSPEAKABLE) != ""


class TagStripper:
    """Removes an optional leading `[emotion] ` from a reply that arrives in pieces.

    Feed each delta in and get back the text that's safe to speak. Text is held only
    while it could still turn out to be a tag, so the latency cost is a few characters.
    """

    def __init__(self) -> None:
        self.tag: str | None = None
        self._held = ""
        self._decided = False

    def feed(self, delta: str) -> str:
        if self._decided:
            return delta
        self._held += delta
        start = self._held.lstrip()
        if not start:
            return ""
        if not start.startswith("["):
            return self._release()
        match = TAG_RE.match(start)
        if match:
            self.tag = match.group(1)
            self._decided, self._held = True, ""
            return start[match.end():]
        if "]" in start or len(start) > MAX_TAG_LENGTH:
            return self._release()   # a bracket, but not a tag: speak it as written
        return ""

    def finish(self) -> str:
        """Whatever is still held when the reply ends."""
        return self._release()

    def _release(self) -> str:
        out, self._held, self._decided = self._held, "", True
        return out
