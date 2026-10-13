"""Dialog-RSN-1's reply, spoken through ElevenLabs while it's still streaming.

    uv run agent.py --say "Tell me about the Moon."   # one text turn, spoken
    uv run agent.py --file call.wav                     # a 16 kHz mono PCM16 WAV as the caller
    uv run agent.py                                     # the microphone: wear headphones

Each section of the streaming to text-to-speech guide has a matching comment below.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
import wave

import sounddevice as sd
import websockets
from dotenv import load_dotenv

from elevenlabs_tts import ElevenLabsTTS
from text import TagStripper, speakable

URL = os.environ.get("DIALOGUE_URL", "wss://api.studio.poly.ai/v1/realtime")
SAMPLE_RATE = 16_000   # Dialog-RSN-1's native input rate
BLOCKSIZE = 512        # 32 ms, the detector's frame size
REPORT_EVERY = 0.25    # seconds between playback reports

INSTRUCTIONS = """\
You are a friendly assistant on a phone call. Keep replies to two or three short sentences.
Choose ONE emotion tag for your response from: [friendly], [apologetic], [reassuring],
[empathetic], [patient]. Return your answer prefixed with the tag in square brackets, for
example "[reassuring] Don't worry, I can get that sorted for you."
"""


class Call:
    def __init__(self, ws, tts: ElevenLabsTTS, verbose: bool = False) -> None:
        self.ws = ws
        self.verbose = verbose
        self.tts = tts
        self.stripper = TagStripper()
        self.spoke = False                    # any text of this response reached TTS
        self.turn_ended_at: float | None = None
        self.first_text_at: float | None = None
        self.started: set[str] = set()        # responses with poly.output_audio.started sent
        self.stopped: set[str] = set()        # ... and .stopped

    async def send(self, event: dict) -> None:
        if self.verbose and event["type"].startswith("poly."):
            shown = {k: v for k, v in event.items() if k not in ("type", "response_id")}
            print(f"\n    -> {event['type']} {json.dumps(shown)}", end="")
        await self.ws.send(json.dumps(event))

    async def handle(self, event: dict) -> bool:
        """Act on one server event. Returns True when a completed response has been spoken."""
        kind = event["type"]

        # One context per response. A new response_id gets a fresh TTS context.
        if kind == "response.created":
            self.tts.start(event["response"]["id"])
            self.stripper, self.spoke, self.first_text_at = TagStripper(), False, None

        # Stream the delta, don't wait for the whole reply.
        elif kind == "response.output_text.delta":
            self.first_text_at = self.first_text_at or time.monotonic()
            await self.speak(self.stripper.feed(event["delta"]))

        elif kind == "response.output_text.done":
            await self.speak(self.stripper.finish())
            if self.spoke:
                await self.tts.flush()
            tag = f"  [{self.stripper.tag}]" if self.stripper.tag else ""
            print(tag if self.spoke else f"(nothing speakable in: {event['text']!r})")

        elif kind == "input_audio_buffer.speech_started":
            await self.interrupt()
            print("(caller speaking)")

        elif kind == "input_audio_buffer.speech_stopped":
            self.turn_ended_at = time.monotonic()

        elif kind == "conversation.item.input_audio_transcription.completed":
            # The model writes its reply before the transcript of what it heard, so this
            # can land while the reply's line is still open.
            print(f"{chr(10) if self.spoke else ''}Caller: {event['transcript'].strip()}")

        elif kind == "response.done":
            status = event["response"]["status"]
            if status == "cancelled":
                await self.interrupt()   # barge-in: the server cancelled this reply
            elif status == "completed":
                return True
            else:
                print(f"[response {status}] {event['response'].get('status_details')}", file=sys.stderr)

        elif kind == "error":
            print(f"[error] {event['error']['code']}: {event['error']['message']}", file=sys.stderr)

        return False

    async def speak(self, text: str) -> None:
        # Guard unspeakable text. Punctuation or whitespace ahead of any real words is
        # dropped, so a reply with nothing to say never reaches the synthesizer at all.
        if not text or not (self.spoke or speakable(text)):
            return
        if not self.spoke:
            print("Agent:  ", end="")
        print(text, end="", flush=True)
        # Strip the emotion tag before it reaches TTS: `text` has already been through
        # the TagStripper, and the tag it found is in self.stripper.tag.
        await self.tts.push(text)
        self.spoke = True

    async def interrupt(self) -> None:
        """Barge-in: cancel and flush are two different steps."""
        if not self.tts.speaking:
            return
        response_id = self.tts.context
        await self.tts.cancel()                   # 1. stop synthesizing this reply
        dropped_ms = self.tts.speaker.flush()     # 2. empty what's already queued to play
        await self.report_stopped(response_id, dropped_ms)
        print(f"\n(interrupted after {self.tts.speaker.played_ms} ms, {dropped_ms} ms dropped)")

    # Report playback back.
    async def report_playback(self) -> None:
        """Runs beside the event loop, turning speaker progress into playback reports."""
        last_ms, last_len = 0, 0
        while True:
            await asyncio.sleep(REPORT_EVERY)
            rid, speaker = self.tts.context, self.tts.speaker
            if rid is None or rid in self.stopped or speaker.played_ms == 0:
                continue
            if rid not in self.started:
                self.started.add(rid)
                last_ms, last_len = 0, 0
                await self.send({"type": "poly.output_audio.started", "response_id": rid})
                self.print_latency()
            spoken = self.tts.spoken_text()
            if speaker.played_ms > last_ms:
                await self.send({"type": "poly.output_audio.delta", "response_id": rid,
                                 "audio_ms": speaker.played_ms, "delta": spoken[last_len:]})
                last_ms, last_len = speaker.played_ms, len(spoken)
            if not self.tts.speaking:
                await self.report_stopped(rid, 0)

    async def report_stopped(self, response_id: str, dropped_ms: int) -> None:
        if response_id in self.started and response_id not in self.stopped:
            self.stopped.add(response_id)
            await self.send({"type": "poly.output_audio.stopped", "response_id": response_id,
                             "audio_ms": self.tts.speaker.played_ms, "dropped_ms": dropped_ms,
                             "spoken_text": self.tts.spoken_text()})

    # Measure time-to-first-audio, not time-to-first-token.
    def print_latency(self) -> None:
        start, first_audio = self.turn_ended_at, self.tts.speaker.started_at
        if start and first_audio and self.first_text_at:
            print(f"\n  (first text {(self.first_text_at - start) * 1000:.0f} ms, "
                  f"first audio {(first_audio - start) * 1000:.0f} ms after the caller finished)")


async def microphone(device):
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[bytes] = asyncio.Queue()
    if device and device.isdigit():
        device = int(device)
    with sd.RawInputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=BLOCKSIZE,
                           device=device,
                           callback=lambda data, *_: loop.call_soon_threadsafe(queue.put_nowait, bytes(data))):
        while True:
            yield await queue.get()


async def wav_file(path):
    """A WAV at real time, then silence, so the call keeps going while the reply plays."""
    with wave.open(path, "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, SAMPLE_RATE):
            sys.exit(f"{path} must be mono, 16-bit, {SAMPLE_RATE} Hz.")
        frames = wav.readframes(wav.getnframes())
    started, step = time.monotonic(), BLOCKSIZE * 2
    silence = bytes(step)
    for i, offset in enumerate(range(0, len(frames) + SAMPLE_RATE * 2 * 12, step)):
        yield frames[offset:offset + step] or silence
        await asyncio.sleep(max(0.0, started + (i + 1) * BLOCKSIZE / SAMPLE_RATE - time.monotonic()))


async def send_audio(ws, chunks):
    async for chunk in chunks:
        await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                  "audio": base64.b64encode(chunk).decode()}))


async def main(args) -> None:
    load_dotenv()
    for var in ("DIALOGUE_API_KEY", "ELEVENLABS_API_KEY"):
        if not os.environ.get(var):
            sys.exit(f"{var} is not set. Copy .env.example to .env and fill it in.")

    tts = ElevenLabsTTS(os.environ["ELEVENLABS_API_KEY"],
                        os.environ.get("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL"),
                        os.environ.get("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5"))
    await tts.connect()
    headers = {"X-API-KEY": os.environ["DIALOGUE_API_KEY"]}
    async with websockets.connect(URL, additional_headers=headers) as ws:
        call = Call(ws, tts, verbose=args.verbose)
        await call.send({"type": "session.update", "session": {
            "instructions": INSTRUCTIONS,
            "output_modalities": ["text"],
            "audio": {"input": {"format": {"type": "audio/pcm"},
                                "turn_detection": {"type": "server_vad", "create_response": True}}},
        }})
        reporter = asyncio.create_task(call.report_playback())

        async def events(stop_after_reply: bool) -> None:
            async for raw in ws:
                if await call.handle(json.loads(raw)) and stop_after_reply:
                    return

        try:
            if args.say:
                print(f"Caller: {args.say}")
                await call.send({"type": "conversation.item.create", "item": {
                    "type": "message", "role": "user",
                    "content": [{"type": "input_text", "text": args.say}]}})
                await call.send({"type": "response.create"})
                call.turn_ended_at = time.monotonic()
                await events(stop_after_reply=True)
            else:
                chunks = wav_file(args.file) if args.file else microphone(args.device)
                tasks = {asyncio.create_task(send_audio(ws, chunks)),
                         asyncio.create_task(events(stop_after_reply=False))}
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                for task in done:
                    task.result()
            await tts.wait_until_spoken()
            await asyncio.sleep(REPORT_EVERY * 2)   # let the last playback report go out
        finally:
            reporter.cancel()
            await tts.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Dialog-RSN-1 with streaming text-to-speech.")
    parser.add_argument("--say", help="send one text turn instead of audio, and exit once it's spoken")
    parser.add_argument("--file", help="stream a 16 kHz mono PCM16 WAV as the caller")
    parser.add_argument("--verbose", action="store_true", help="print the playback reports as they're sent")
    parser.add_argument("--device", help="microphone index or name, see `python -m sounddevice`")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        asyncio.run(main(parse_args()))
    except KeyboardInterrupt:
        print("\nbye")
