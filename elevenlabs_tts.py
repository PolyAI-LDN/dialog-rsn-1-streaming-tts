"""ElevenLabs over one websocket for the whole call, one context per response.

The `multi-stream-input` endpoint keeps a single socket open and multiplexes replies by
`context_id`, so a new turn doesn't pay for a handshake before its first audio. Text goes
in as it arrives from the model, and PCM comes back and goes straight to the speaker.
Character timings from ElevenLabs are what `spoken_text` is built from.
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys

import websockets

from speaker import Speaker

RATE = 24_000
INACTIVITY_TIMEOUT = 180   # seconds, the most ElevenLabs allows


class ElevenLabsTTS:
    def __init__(self, api_key: str, voice_id: str, model_id: str) -> None:
        self.api_key = api_key
        self.url = (f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/multi-stream-input"
                    f"?model_id={model_id}&output_format=pcm_{RATE}"
                    f"&inactivity_timeout={INACTIVITY_TIMEOUT}&sync_alignment=true")
        self.speaker = Speaker(RATE)
        self._ws = None
        self._reader: asyncio.Task | None = None
        self.context: str | None = None    # the response being spoken
        self._opened = False               # first text sent for this context
        self._closed = False               # close_context sent
        self._cancelled = False
        self._final = asyncio.Event()      # ElevenLabs has sent the last audio for it
        self._chars: list[tuple[int, str]] = []   # (start ms within the reply, character)
        self._received_ms = 0

    async def connect(self) -> None:
        self._ws = await websockets.connect(self.url, max_size=None,
                                            additional_headers={"xi-api-key": self.api_key})
        self._reader = asyncio.create_task(self._read())

    async def close(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.send(json.dumps({"close_socket": True}))
            except websockets.ConnectionClosed:
                pass
            await self._ws.close()
        if self._reader is not None:
            self._reader.cancel()
        self.speaker.close()

    def start(self, response_id: str) -> None:
        """A new context for a new response. Nothing is sent until its first text."""
        self.context = response_id
        self._opened = self._closed = self._cancelled = False
        self._final.clear()
        self._chars, self._received_ms = [], 0
        self.speaker.reset()

    async def push(self, text: str) -> None:
        """Feed the increment, not the reply so far."""
        if self._cancelled:
            return
        if not self._opened:
            await self._send({"text": " ", "context_id": self.context})
            self._opened = True
        await self._send({"text": text, "context_id": self.context})

    async def flush(self) -> None:
        """No more text for this response.

        `flush` makes ElevenLabs synthesize what it's still holding, and `close_context`
        ends the context once that audio is out, which is what brings `isFinal`. Closing
        without flushing first would drop the end of the reply.
        """
        if self._opened and not self._cancelled:
            await self._send({"context_id": self.context, "flush": True})
            await self._close_context()

    async def cancel(self) -> None:
        """Stop synthesizing this response. Audio already in the speaker isn't touched."""
        if self._cancelled:
            return
        self._cancelled = True
        self._final.set()
        if self._opened:
            await self._close_context()   # without a flush, so unsynthesized text is dropped

    async def _close_context(self) -> None:
        if not self._closed:
            self._closed = True
            await self._send({"context_id": self.context, "close_context": True})

    async def wait_until_spoken(self) -> None:
        """Until the last audio for this response has played."""
        if not self._opened:
            return
        await self._final.wait()
        while not self.speaker.finished:
            await asyncio.sleep(0.02)

    @property
    def speaking(self) -> bool:
        return self._opened and not self._cancelled and not (
            self._final.is_set() and self.speaker.finished)

    def spoken_text(self) -> str:
        """The characters whose audio has actually played."""
        played = self.speaker.played_ms
        return "".join(ch for ms, ch in self._chars if ms <= played)

    async def _send(self, message: dict) -> None:
        try:
            await self._ws.send(json.dumps(message))
        except websockets.ConnectionClosed as exc:
            print(f"[tts] socket closed: {exc}", file=sys.stderr)

    async def _read(self) -> None:
        async for raw in self._ws:
            msg = json.loads(raw)
            if msg.get("error"):
                print(f"[tts] {msg}", file=sys.stderr)
                continue
            if msg.get("contextId") != self.context or self._cancelled:
                continue   # late audio for a response that's been cancelled or replaced
            if msg.get("audio"):
                pcm = base64.b64decode(msg["audio"])
                alignment = msg.get("alignment") or msg.get("normalizedAlignment")
                if alignment:
                    # Timings count from the start of this context, not of each chunk.
                    self._chars.extend(zip(alignment["charStartTimesMs"], alignment["chars"]))
                self._received_ms += len(pcm) // (RATE * 2 // 1000)
                self.speaker.enqueue(pcm)
            if msg.get("isFinal"):
                if self._received_ms == 0:
                    # This endpoint reports some refusals, such as a voice your plan can't
                    # use, only as a final message with no audio. Say so rather than go quiet.
                    print("[tts] ElevenLabs sent no audio for this reply. Check the voice ID "
                          "and your plan: the REST endpoint shows the reason.", file=sys.stderr)
                self.speaker.mark_end()
                self._final.set()
