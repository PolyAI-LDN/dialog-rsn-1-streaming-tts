"""The speaker: an output stream fed from a queue that can be emptied.

Cancelling synthesis doesn't reach audio already queued here. `flush` does. `played_ms`
counts only audio handed to the device, so playback reports say what the caller heard
rather than what was synthesized.
"""

from __future__ import annotations

import threading
import time
from collections import deque

import sounddevice as sd

BLOCK_MS = 20


class Speaker:
    def __init__(self, rate: int) -> None:
        self.rate = rate
        self._bytes_per_ms = rate * 2 // 1000
        self._queue: deque[bytes] = deque()
        self._lock = threading.Lock()
        self._played = 0          # bytes handed to the device for the current reply
        self._queued = 0          # bytes waiting
        self._ended = False       # the last chunk of the current reply is in
        self.started_at: float | None = None   # monotonic time of the first played sample
        self._stream = sd.RawOutputStream(samplerate=rate, channels=1, dtype="int16",
                                          blocksize=rate * BLOCK_MS // 1000, callback=self._fill)
        self._stream.start()

    def reset(self) -> None:
        """Start counting for a new reply."""
        with self._lock:
            self._queue.clear()
            self._played = self._queued = 0
            self._ended = False
            self.started_at = None

    def enqueue(self, pcm16: bytes) -> None:
        with self._lock:
            self._queue.append(pcm16)
            self._queued += len(pcm16)

    def mark_end(self) -> None:
        with self._lock:
            self._ended = True

    def flush(self) -> int:
        """Drop everything not yet played. Returns the milliseconds dropped."""
        with self._lock:
            dropped = self._queued
            self._queue.clear()
            self._queued = 0
            self._ended = True
            return dropped // self._bytes_per_ms

    @property
    def played_ms(self) -> int:
        return self._played // self._bytes_per_ms

    @property
    def finished(self) -> bool:
        with self._lock:
            return self._ended and self._queued == 0

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()

    def _fill(self, outdata, frames, time_info, status) -> None:
        need = len(outdata)
        chunks, got = [], 0
        with self._lock:
            while got < need and self._queue:
                head = self._queue[0]
                take = min(need - got, len(head))
                chunks.append(head[:take])
                got += take
                if take == len(head):
                    self._queue.popleft()
                else:
                    self._queue[0] = head[take:]
            self._queued -= got
            self._played += got
            if got and self.started_at is None:
                self.started_at = time.monotonic()
        outdata[:got] = b"".join(chunks)
        outdata[got:] = bytes(need - got)
