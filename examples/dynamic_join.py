#!/usr/bin/env python3
"""Dynamic join/leave demo: requests enter and exit the batch mid-generation.

This demonstrates the core vTTS capability — continuous batching where
new requests join the generation loop without waiting for existing ones
to finish.
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

    texts = [
        ("Good morning, welcome to our service.", "vivian", "English"),
        ("Please hold while I check your account.", "ryan", "English"),
        ("Your order has been confirmed successfully.", "serena", "English"),
        ("Thank you for calling, have a great day.", "eric", "English"),
    ]

    requests = []
    t0 = time.time()

    # Wave 1: Start 2 requests
    for i in range(2):
        text, speaker, lang = texts[i]
        req = TTSRequest(request_id=f"req_{i}", text=text, voice_id=speaker, language=lang)
        worker.prefill_slot_speaker(req)
        requests.append(req)
    print(f"[{time.time()-t0:.2f}s] Wave 1: {worker.num_active_slots} slots active")

    # Run 20 steps
    for _ in range(20):
        worker.step()

    # Wave 2: 2 more requests JOIN the batch mid-generation
    for i in range(2, 4):
        text, speaker, lang = texts[i]
        req = TTSRequest(request_id=f"req_{i}", text=text, voice_id=speaker, language=lang)
        worker.prefill_slot_speaker(req)
        requests.append(req)
    print(f"[{time.time()-t0:.2f}s] Wave 2: {worker.num_active_slots} slots active (dynamic join!)")

    # Run until all done
    step = 20
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
            dur = len(audio) / req.sample_rate
            total_audio += dur
            fname = f"output_dyn_{texts[i][1]}.wav"
            sf.write(fname, audio, req.sample_rate)
            ttfb = (req.first_chunk_at - t0) * 1000 if req.first_chunk_at else -1
            print(f"  {texts[i][1]:8s}: {dur:.1f}s  TTFB={ttfb:.0f}ms  -> {fname}")

    print(f"\nTotal: {total_audio:.1f}s audio in {elapsed:.1f}s "
          f"({total_audio/elapsed:.1f} aud_s/s)")


if __name__ == "__main__":
    main()
