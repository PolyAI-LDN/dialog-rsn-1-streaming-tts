# Streaming Dialog-RSN-1 replies to text-to-speech

Companion code for [Streaming text to your text-to-speech](https://dialog-rsn-1-eap-docs.pages.dev/dialog-rsn-1/guides/streaming-text-to-speech/).

Dialog-RSN-1 answers in text. This agent speaks each reply through
[ElevenLabs](https://elevenlabs.io) while the text is still arriving, stops cleanly when the
caller talks over it, and tells the server what the caller actually heard.

```text
$ uv run agent.py --file barge-in.wav
Agent:  The Apollo program led by NASA put humans on the moon six times between 1969 and 1972. ...
  (first text 214 ms, first audio 850 ms after the caller finished)
(interrupted after 5480 ms, 9729 ms dropped)
Caller: Sorry, stop there. Actually, how far away is the moon?
Agent:  The moon is about 384,400 kilometers away from Earth on average. ...
```

## Where each part of the guide lives

| Guide section | Code |
|---|---|
| Stream the delta, don't wait for the whole reply | `agent.py`: `response.output_text.delta` goes straight to `tts.push()` |
| One context per response | `elevenlabs_tts.py`: one socket for the call, one ElevenLabs context per `response_id` |
| Cancel and flush are two different steps | `agent.py`: `interrupt()` cancels synthesis, then empties the speaker |
| Guard unspeakable text | `text.py`: `speakable()`, used by `agent.py`'s `speak()` |
| Strip the emotion tag | `text.py`: `TagStripper`, which works on a reply that arrives in pieces |
| Report playback back | `agent.py`: `report_playback()` sends `poly.output_audio.*` every 250 ms |
| Measure time-to-first-audio | `agent.py`: prints first text and first audio, both timed from the end of the caller's turn |

`speaker.py` is the output device: a queue that can be emptied, which counts only the audio
it has actually played.

## Run it

You need Python 3.12, [uv](https://docs.astral.sh/uv/), a Dialog-RSN-1 key and an ElevenLabs
API key. ElevenLabs' free tier works.

```bash
cp .env.example .env    # add DIALOG_API_KEY and ELEVENLABS_API_KEY
uv sync

uv run agent.py --say "Tell me three facts about the Moon."   # one text turn, spoken
uv run agent.py                                               # the microphone
```

**Wear headphones with the microphone.** Through speakers, the mic hears the agent's voice
and treats it as the caller interrupting.

Add `--verbose` to print each playback report as it's sent.

### Try an interruption without a microphone

`--file` streams a WAV as the caller, so you can test barge-in on speakers. This one asks a
question, leaves six seconds for the reply to start, then talks over it:

```bash
say -o q1.aiff "Can you tell me a bit about the history of the Moon landings?"
say -o q2.aiff "Sorry, stop there. Actually, how far away is the Moon?"
afconvert -f WAVE -d LEI16@16000 -c 1 q1.aiff q1.wav
afconvert -f WAVE -d LEI16@16000 -c 1 q2.aiff q2.wav
uv run python -c "
import wave
parts = [wave.open(p).readframes(10**9) for p in ('q1.wav', 'q2.wav')]
with wave.open('barge-in.wav', 'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(parts[0] + bytes(16000 * 2 * 6) + parts[1])
"
uv run agent.py --verbose --file barge-in.wav
```

Look for `poly.output_audio.stopped` with `dropped_ms` above zero and a `spoken_text` that
ends where the interruption began.

## ElevenLabs details that matter

- **A reply ends with `flush`, then `close_context`.** `flush` makes ElevenLabs synthesize the
  text it's still holding, but only `close_context` ends the context and brings `isFinal`.
  `close_context` on its own drops that held text, which is what a barge-in wants.
- **Character timings count from the start of the context,** not from each audio chunk.
  `spoken_text` compares them with how much audio has played.
- **Free accounts can't use library voices through the API.** The default voice is Sarah
  (`EXAVITQu4vr4xnSDxMaL`), one of the default voices free accounts can use. Set
  `ELEVENLABS_VOICE_ID` to change it. The multi-context endpoint reports a voice your plan
  can't use only as a final message with no audio, so the agent prints a warning when that
  happens.

## License

Apache 2.0. See [LICENSE](LICENSE).
