#!/usr/bin/env python3
"""Generate speech with multiple voices simultaneously.

All voices are processed in a single batched forward pass per step.
"""

import time

import numpy as np
import soundfile as sf

from vtts import ContinuousWorker, TTSRequest


def main():
    worker = ContinuousWorker(
        model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device="cuda:0",
        emit_every_frames=6,
    )
    worker.load_model()

    # Define texts and speakers
    items = [
        ("Good morning, how can I help you today?", "vivian", "English"),
        ("The weather forecast says it will rain tomorrow.", "ryan", "English"),
        ("Buenos dias, bienvenido al sistema.", "serena", "Spanish"),
        ("Technology keeps evolving at an incredible pace.", "eric", "English"),
    ]

    # Start all generations
    requests = []
    t0 = time.time()

    for text, speaker, lang in items:
        req = TTSRequest(
            request_id=speaker,
            text=text,
            voice_id=speaker,
            language=lang,
        )
        worker.prefill_slot_speaker(req)
        requests.append(req)

    print(f"All {len(requests)} slots prefilled in {time.time()-t0:.2f}s")

    # Run until all done (single batched forward pass per step)
    step = 0
    while worker.num_active_slots > 0:
        worker.step()
        step += 1

    elapsed = time.time() - t0
    total_audio = 0

    for i, req in enumerate(requests):
        chunks = []
        while not req.chunk_queue.empty():
            c = req.chunk_queue.get_nowait()
            if c is not None:
                chunks.append(c)
        if chunks:
            audio = np.concatenate(chunks)
            duration = len(audio) / req.sample_rate
            total_audio += duration
            fname = f"output_{items[i][1]}.wav"
            sf.write(fname, audio, req.sample_rate)
            print(f"  {items[i][1]:8s}: {duration:.1f}s -> {fname}")

    print(f"\nTotal audio: {total_audio:.1f}s in {elapsed:.1f}s "
          f"(throughput: {total_audio/elapsed:.1f} aud_s/s)")


if __name__ == "__main__":
    main()
