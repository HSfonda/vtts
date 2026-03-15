# vTTS - Continuous Batching for Text-to-Speech
# https://github.com/caimari/vtts
# Copyright (c) 2025 Antoni Caimari Caldes
# Licensed under MIT License

"""
FastAPI server for vTTS continuous batching.

Endpoints:
  POST /v1/tts/generate     — Generate audio (SSE stream of base64 PCM chunks)
  POST /v1/voices/register  — Register a cloned voice (Base models only)
  GET  /v1/voices            — List available voices
  GET  /v1/stats             — Server statistics
  GET  /health               — Health check
"""

import asyncio
import base64
import json
import logging
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from vtts.config import VTTSConfig
from vtts.worker import ContinuousWorker, TTSRequest
from vtts.scheduler import ContinuousScheduler
from vtts.voice_registry import VoiceRegistry

logger = logging.getLogger("vtts.server")

app = FastAPI(title="vTTS", version="0.1.0",
              description="Virtual TTS: continuous batching for text-to-speech")


# Global state (initialized in startup)
worker: Optional[ContinuousWorker] = None
scheduler: Optional[ContinuousScheduler] = None
registry: Optional[VoiceRegistry] = None
config: Optional[VTTSConfig] = None


class GenerateRequest(BaseModel):
    text: str
    voice_id: str
    language: str = "English"
    request_id: Optional[str] = None


class RegisterVoiceRequest(BaseModel):
    ref_audio_path: str
    ref_text: str
    voice_id: Optional[str] = None
    name: Optional[str] = None


def create_app(cfg: VTTSConfig) -> FastAPI:
    """Create and configure the FastAPI app."""
    global config
    config = cfg

    @app.on_event("startup")
    async def startup():
        global worker, scheduler, registry

        w = ContinuousWorker(
            model_name=cfg.model_path,
            device=cfg.device,
            emit_every_frames=cfg.emit_every_frames,
            decode_window_frames=cfg.decode_window_frames,
            overlap_samples=cfg.overlap_samples,
            max_frames=cfg.max_frames,
            max_slots=cfg.max_slots,
        )
        w.load_model()

        loop = asyncio.get_event_loop()
        w.set_event_loop(loop)

        reg = VoiceRegistry()
        for vid, vcfg in cfg.voices.items():
            reg.register(
                model=w.model,
                ref_audio_path=vcfg.ref_audio,
                ref_text=vcfg.ref_text,
                voice_id=vid,
            )

        sched = ContinuousScheduler(w, voice_registry=reg)
        asyncio.create_task(sched.start())

        worker = w
        scheduler = sched
        registry = reg
        logger.info("vTTS server ready")

    return app


@app.post("/v1/tts/generate")
async def generate(req: GenerateRequest):
    """Generate speech. Returns SSE stream of base64-encoded PCM chunks."""
    if scheduler is None:
        raise HTTPException(503, "Server not ready")

    request_id = req.request_id or str(uuid.uuid4())[:8]
    tts_req = TTSRequest(
        request_id=request_id,
        text=req.text,
        voice_id=req.voice_id,
        language=req.language,
    )

    await scheduler.submit(tts_req)

    async def stream():
        while True:
            chunk = await tts_req.chunk_queue.get()
            if chunk is None:
                yield f"data: {json.dumps({'done': True, 'request_id': request_id})}\n\n"
                break
            audio_b64 = base64.b64encode(chunk.astype(np.float32).tobytes()).decode()
            yield f"data: {json.dumps({'audio': audio_b64, 'sr': tts_req.sample_rate})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/v1/voices/register")
async def register_voice(req: RegisterVoiceRequest):
    """Register a cloned voice (Base models only)."""
    if registry is None or worker is None:
        raise HTTPException(503, "Server not ready")
    try:
        vid = registry.register(
            model=worker.model,
            ref_audio_path=req.ref_audio_path,
            ref_text=req.ref_text,
            voice_id=req.voice_id,
            name=req.name,
        )
        return {"voice_id": vid, "status": "registered"}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/v1/voices")
async def list_voices():
    """List all available voices (built-in + cloned)."""
    voices = []
    if worker:
        for sp in worker.get_supported_speakers():
            voices.append({"voice_id": sp, "type": "built-in"})
    if registry:
        for v in registry.list_voices():
            v["type"] = "cloned"
            voices.append(v)
    return {"voices": voices}


@app.get("/v1/stats")
async def stats():
    return {
        "active_slots": worker.num_active_slots if worker else 0,
        "max_slots": config.max_slots if config else 0,
        "model": config.model_name if config else None,
        "device": config.device if config else None,
    }


@app.get("/health")
async def health():
    return {
        "status": "ok" if worker else "loading",
        "mode": "continuous_batching",
        "version": "0.1.0",
    }
