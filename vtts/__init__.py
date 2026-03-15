# vTTS - Continuous Batching for Text-to-Speech
# https://github.com/caimari/vtts
# Copyright (c) 2025 Antoni Caimari Caldes
# Licensed under MIT License

"""vTTS — Virtual Text-to-Speech: continuous batching for TTS models.

Like vLLM virtualizes LLM inference for multiple users on a single GPU,
vTTS virtualizes TTS inference with continuous batching, dynamic join/leave,
and batched forward passes.
"""

__version__ = "0.1.0"

from vtts.config import VTTSConfig
from vtts.worker import ContinuousWorker, TTSRequest
from vtts.voice_registry import VoiceRegistry

__all__ = [
    "VTTSConfig",
    "ContinuousWorker",
    "TTSRequest",
    "VoiceRegistry",
]
