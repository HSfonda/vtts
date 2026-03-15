#!/usr/bin/env python3
"""Basic example: generate speech with a built-in speaker."""

import numpy as np
import soundfile as sf

from vtts import ContinuousWorker, TTSRequest


def main():
    # Load model (downloads automatically from HuggingFace on first run)
    worker = ContinuousWorker(
        model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device="cuda:0",
        emit_every_frames=6,
    )
    worker.load_model()

    # List available speakers
    speakers = worker.get_supported_speakers()
    print(f"Available speakers: {speakers}")

    # Generate
    req = TTSRequest(
        request_id="example",
        text="Hello! This is a demonstration of the vTTS system generating speech.",
        voice_id="vivian",
        language="English",
    )
    worker.prefill_slot_speaker(req)

    while worker.num_active_slots > 0:
        worker.step()

    # Collect and save audio
    chunks = []
    while not req.chunk_queue.empty():
        chunk = req.chunk_queue.get_nowait()
        if chunk is not None:
            chunks.append(chunk)

    audio = np.concatenate(chunks)
    sf.write("output_basic.wav", audio, req.sample_rate)
    print(f"Saved: output_basic.wav ({len(audio)/req.sample_rate:.1f}s)")


if __name__ == "__main__":
    main()
