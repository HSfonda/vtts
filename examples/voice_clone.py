#!/usr/bin/env python3
"""Voice cloning example: clone a voice from a reference audio.

Uses a Base model (not CustomVoice) to clone from a reference WAV file.

Requirements:
  - A reference audio file (~5-10 seconds of clean speech)
  - The exact transcript of what is said in the audio
"""

import sys

import numpy as np
import soundfile as sf

from vtts import ContinuousWorker, TTSRequest, VoiceRegistry


def main():
    # Check arguments
    if len(sys.argv) < 3:
        print("Usage: python voice_clone.py <ref_audio.wav> <ref_text>")
        print()
        print("Example:")
        print('  python voice_clone.py ./my_voice.wav "The exact words in the audio."')
        print()
        print("The reference audio should be ~5-10 seconds of clean speech.")
        print("The ref_text must match exactly what is said in the audio.")
        sys.exit(1)

    ref_audio = sys.argv[1]
    ref_text = sys.argv[2]

    # Load a Base model (supports voice cloning)
    worker = ContinuousWorker(
        model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        device="cuda:0",
        emit_every_frames=6,
    )
    worker.load_model()

    # Register the voice
    registry = VoiceRegistry()
    registry.register(
        model=worker.model,
        ref_audio_path=ref_audio,
        ref_text=ref_text,
        voice_id="cloned",
    )
    print(f"Voice registered from: {ref_audio}")

    # Generate speech with the cloned voice
    text = "This is a test of voice cloning. The generated audio should sound like the reference."
    req = TTSRequest(
        request_id="clone_test",
        text=text,
        voice_id="cloned",
        language="English",
    )
    worker.prefill_slot(req, registry)

    while worker.num_active_slots > 0:
        worker.step()

    chunks = []
    while not req.chunk_queue.empty():
        c = req.chunk_queue.get_nowait()
        if c is not None:
            chunks.append(c)

    if chunks:
        audio = np.concatenate(chunks)
        sf.write("output_cloned.wav", audio, req.sample_rate)
        print(f"Saved: output_cloned.wav ({len(audio)/req.sample_rate:.1f}s)")
    else:
        print("Error: no audio generated")


if __name__ == "__main__":
    main()
