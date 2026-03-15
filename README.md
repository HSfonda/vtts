# vTTS — Virtual Text-to-Speech

**Continuous batching for TTS models. Like vLLM, but for voice.**

[Documentacion en espanol](README.es.md)

vTTS serves multiple text-to-speech requests simultaneously on a single GPU using continuous batching with dynamic join/leave — the same technique that vLLM uses for LLMs, applied to speech generation.

## Key Results

| Metric | Value |
|---|---|
| Simultaneous voices on one GPU | **8-10** (RTX 3060 12GB) |
| Time to first audio byte (1 user) | **192ms** (RTX 3090 Ti) / **254ms** (RTX 3060) |
| Time to first audio byte (4 users) | **254ms** (RTX 3090 Ti) / **382ms** (RTX 3060) |
| Time to first audio byte (8 users) | **368ms** (RTX 3090 Ti) / **570ms** (RTX 3060) |
| Extra VRAM per user | **~2.5MB** (KV cache only) |
| Throughput (5 simultaneous) | **3.6 audio seconds/wall second** |

These numbers were measured with the Qwen3-TTS-12Hz-1.7B-CustomVoice model (3.4GB VRAM).

## How It Works

Traditional TTS servers process one request at a time — each user locks the GPU for the entire generation. vTTS changes this by running all active requests in a **single batched forward pass** each step, with requests joining and leaving the batch dynamically.

### Architecture Modes

#### M1 — Batch Epochs (simple)

All requests in a batch start and finish together. New requests wait until the current batch completes. Simple, efficient, but adds latency for new arrivals.

Best for: **batch dubbing**, offline processing, scenarios where all texts are known upfront.

```
Batch 1: [req_A, req_B, req_C] → all generate together → all finish
         (new requests wait here)
Batch 2: [req_D, req_E] → all generate together → all finish
```

#### M2 — Continuous Batching (advanced)

Requests join and leave the generation loop at any time. A new request enters the batch within one step (~80ms), without waiting for others to finish. This is the **vLLM-equivalent** for TTS.

Best for: **real-time voice agents**, streaming APIs, any scenario where requests arrive unpredictably.

```
Step 1:  [req_A, req_B]           ← A and B generating
Step 10: [req_A, req_B, req_C]    ← C joins mid-generation (dynamic join)
Step 25: [req_A, req_C]           ← B finished and left (dynamic leave)
Step 30: [req_A, req_C, req_D]    ← D joins
Step 40: [req_C, req_D]           ← A finished
```

**How the batched forward works internally:**

1. Each slot has its own KV cache (different lengths) and generation step
2. KV caches are padded to the same length with attention_mask=0
3. Per-slot text conditioning is applied before batching
4. One forward pass through the transformer processes all slots
5. After the forward, KV caches are extracted back per-slot (padding stripped)

## Supported Models

| Model | Size | VRAM | Mode | Speakers |
|---|---|---|---|---|
| [Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) | 0.6B | ~1.2GB | Voice cloning | Any (from reference audio) |
| [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | 1.7B | ~3.4GB | Voice cloning | Any (from reference audio) |
| [Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) | 0.6B | ~1.2GB | Built-in speakers | aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian |
| [Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | 1.7B | ~3.4GB | Built-in speakers | aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian |

Models are downloaded automatically from HuggingFace on first use. If you already have a model downloaded locally, you can point to it in the config (see [Configuration](#configuration)).

## Requirements

- Python >= 3.10
- CUDA GPU (tested on RTX 3060 12GB, RTX 3090 Ti 24GB)
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — the Qwen3-TTS inference library

## Installation

```bash
git clone https://github.com/caimari/vtts.git
cd vtts
pip install -e .
```

You also need the Qwen3-TTS inference library:

```bash
git clone https://github.com/QwenLM/Qwen3-TTS.git
pip install -e Qwen3-TTS/
```

## Quick Start

### Built-in Speakers (CustomVoice models)

```python
from vtts import ContinuousWorker, TTSRequest
import numpy as np
import soundfile as sf

# Load model (downloaded automatically from HuggingFace)
worker = ContinuousWorker(
    model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device="cuda:0",
    emit_every_frames=6,   # ~500ms chunks
)
worker.load_model()

# Generate speech
req = TTSRequest(
    request_id="hello",
    text="Hello, this is a test of the vTTS system.",
    voice_id="vivian",      # Built-in speaker
    language="English",
)
worker.prefill_slot_speaker(req)

while worker.num_active_slots > 0:
    worker.step()

# Collect audio
chunks = []
while not req.chunk_queue.empty():
    chunk = req.chunk_queue.get_nowait()
    if chunk is not None:
        chunks.append(chunk)

audio = np.concatenate(chunks)
sf.write("output.wav", audio, req.sample_rate)
```

### Voice Cloning (Base models)

```python
from vtts import ContinuousWorker, TTSRequest, VoiceRegistry

worker = ContinuousWorker(
    model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device="cuda:0",
)
worker.load_model()

# Register a voice from a reference audio
# The audio should be ~5-10 seconds of clean speech.
# ref_text must be the EXACT transcript of what is said in the audio.
registry = VoiceRegistry()
registry.register(
    model=worker.model,
    ref_audio_path="./voices/my_voice.wav",
    ref_text="The exact words spoken in the reference audio file.",
    voice_id="my_voice",
)

req = TTSRequest(
    request_id="clone_test",
    text="This will sound like the reference voice.",
    voice_id="my_voice",
    language="English",
)
worker.prefill_slot(req, registry)

while worker.num_active_slots > 0:
    worker.step()
```

### Multiple Simultaneous Voices

```python
# All of these generate in a SINGLE batched forward pass per step
requests = []
for speaker in ["vivian", "ryan", "serena", "eric"]:
    req = TTSRequest(
        request_id=speaker,
        text=f"Hello from {speaker}!",
        voice_id=speaker,
        language="English",
    )
    worker.prefill_slot_speaker(req)
    requests.append(req)

# One loop drives all 4 generations simultaneously
while worker.num_active_slots > 0:
    worker.step()
```

## Voice Cloning: Reference Audio Guidelines

For Base models (voice cloning), you need a reference audio file:

- **Duration**: 5-10 seconds (shorter is better than too long)
- **Content**: Clear speech, minimal background noise
- **Format**: WAV (mono or stereo, any sample rate)
- **ref_text**: Must be the **exact transcript** of what is said in the audio. If the transcript doesn't match the audio, quality will degrade significantly
- **Language**: The reference audio language doesn't need to match the target language

## Configuration

Copy `config.example.yaml` to `config.yaml`:

```bash
cp config.example.yaml config.yaml
```

Key settings:

```yaml
model:
  name: "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
  device: "cuda:0"

  # If you already downloaded the model, point to it here
  # to avoid re-downloading:
  # local_path: "/path/to/your/downloaded/model"

worker:
  emit_every_frames: 6    # Lower = lower latency
  max_slots: 16           # Max concurrent generations
  max_frames: 3600        # Max audio length (12 frames = 1 second)
```

### Using a Pre-Downloaded Model

If you already have a Qwen3-TTS model on disk (e.g., from a previous HuggingFace download), set `local_path` in your config:

```yaml
model:
  name: "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
  local_path: "/home/user/models/Qwen3-TTS-12Hz-1.7B-CustomVoice"
```

Or pass it directly in code:

```python
worker = ContinuousWorker(
    model_name="/home/user/models/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device="cuda:0",
)
```

## API Server

```bash
# Start the server
python -m vtts.server --config config.yaml

# Or with uvicorn directly
uvicorn vtts.server:app --host 0.0.0.0 --port 8080
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/v1/tts/generate` | Generate speech (SSE stream) |
| POST | `/v1/voices/register` | Register a cloned voice |
| GET | `/v1/voices` | List available voices |
| GET | `/v1/stats` | Server statistics |
| GET | `/health` | Health check |

### Generate Speech

```bash
curl -X POST http://localhost:8080/v1/tts/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice_id": "vivian", "language": "English"}'
```

## Latency Tuning

The `emit_every_frames` parameter controls the trade-off between latency and efficiency:

| emit_every_frames | Chunk size | TTFB (1 user) | Best for |
|---|---|---|---|
| 3 | ~250ms | ~200ms | Voice agents, real-time |
| 6 | ~500ms | ~400ms | Streaming APIs |
| 20 | ~1.7s | ~1.4s | Batch processing |

For voice agent applications, use `emit_every_frames=3` for sub-300ms time-to-first-byte.

## VRAM Budget

| Component | VRAM |
|---|---|
| 0.6B model weights (bf16) | ~1.2 GB |
| 1.7B model weights (bf16) | ~3.4 GB |
| Vocoder weights | ~0.8 GB |
| KV cache per user (~200 frames) | ~2.5 MB |
| CUDA overhead | ~0.5 GB |

With 10 simultaneous users on a 1.7B model: ~4.7 GB total. Fits comfortably on a 12GB GPU.

## Using vTTS in a Voice Agent Pipeline

vTTS is designed to be used as the TTS component in a real-time voice agent pipeline. A typical architecture looks like this:

```
User (phone/browser)
    │
    ▼
SIP / WebRTC gateway
    │
    ▼
STT (Speech-to-Text)  ──►  Transcribed text
    │
    ▼
LLM (Language Model)   ──►  Response text
    │
    ▼
vTTS (this library)    ──►  Audio chunks (streaming)
    │
    ▼
User hears the response
```

Your orchestrator connects these components. vTTS fits in as a library — no separate server required:

```python
from vtts import ContinuousWorker, TTSRequest

# Initialize once at startup
worker = ContinuousWorker(
    model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device="cuda:0",
    emit_every_frames=3,  # Low latency for voice agents
)
worker.load_model()

# In your orchestrator's request handler:
async def handle_call(user_id, llm_response_text):
    req = TTSRequest(
        request_id=user_id,
        text=llm_response_text,
        voice_id="vivian",
        language="English",
    )
    worker.prefill_slot_speaker(req)

    # Stream audio chunks back to the user as they are generated
    while True:
        chunk = await req.chunk_queue.get()
        if chunk is None:
            break
        send_audio_to_user(user_id, chunk, sample_rate=req.sample_rate)
```

The key advantage of vTTS here is that **multiple calls generate simultaneously** in a single batched forward pass. If 10 users are in calls at the same time, the GPU processes all 10 in each step — no queuing, no waiting.

For the STT side, most engines (Whisper, Deepgram, etc.) already support concurrent requests natively and don't require custom batching.

## Why Not Just Run Multiple Processes?

You could run N copies of the model (one per user), but:

| Approach | Users | VRAM | Speed |
|---|---|---|---|
| 3 separate processes | 3 | 7.3 GB each (22 GB total) | 43s per generation |
| vTTS continuous batching | 10 | 3.4 GB total (+2.5 MB/user) | 3.6 audio sec/wall sec |

Multiple processes duplicate model weights in VRAM. vTTS shares one model across all users, batching their forward passes together. On a 12GB GPU, multiprocess maxes out at 3 users. vTTS serves 10+ with room to spare.

## Author

Created by [Antoni Caimari Caldes](https://github.com/caimari) — acaimari22@gmail.com

## License

MIT — see [LICENSE](LICENSE).
