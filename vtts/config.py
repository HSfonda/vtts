# vTTS - Continuous Batching for Text-to-Speech
# https://github.com/caimari/vtts
# Copyright (c) 2025 Antoni Caimari Caldes
# Licensed under MIT License

"""Configuration management for vTTS."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import yaml


@dataclass
class VoiceConfig:
    """Configuration for a cloned voice."""
    ref_audio: str
    ref_text: str


@dataclass
class VTTSConfig:
    """Main configuration for vTTS server and worker.

    Can be loaded from a YAML file or constructed directly.

    Args:
        model_name: HuggingFace model ID or local path.
            Supported models:
            - "Qwen/Qwen3-TTS-12Hz-0.6B-Base" (voice cloning, smallest)
            - "Qwen/Qwen3-TTS-12Hz-1.7B-Base" (voice cloning, larger)
            - "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice" (built-in speakers)
            - "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" (built-in speakers, larger)
        device: CUDA device (e.g., "cuda:0", "cuda:1").
        dtype: Model precision ("bfloat16" or "float16").
        max_slots: Maximum concurrent generations.
        emit_every_frames: Frames between audio chunk emissions.
            Lower = lower latency, higher overhead. Default 6 (~500ms chunks).
        decode_window_frames: Vocoder context window size.
        overlap_samples: Crossfade overlap between chunks.
        max_frames: Maximum generation length (12 frames = 1 second).
        host: Server bind address.
        port: Server port.
        voices: Dict of voice_id -> VoiceConfig for voice cloning.
        model_local_path: Optional local path if model is already downloaded.
            If set, this path is used instead of downloading from HuggingFace.
    """
    # Model
    model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    model_local_path: Optional[str] = None

    # Worker
    max_slots: int = 16
    emit_every_frames: int = 6
    decode_window_frames: int = 80
    overlap_samples: int = 1024
    max_frames: int = 3600

    # Server
    host: str = "0.0.0.0"
    port: int = 8080

    # Voices (for voice cloning models)
    voices: Dict[str, VoiceConfig] = field(default_factory=dict)

    @property
    def model_path(self) -> str:
        """Return local path if set, otherwise HuggingFace model ID."""
        if self.model_local_path and os.path.isdir(self.model_local_path):
            return self.model_local_path
        return self.model_name

    @classmethod
    def from_yaml(cls, path: str) -> "VTTSConfig":
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        model_cfg = data.get("model", {})
        server_cfg = data.get("server", {})
        worker_cfg = data.get("worker", {})
        voices_raw = data.get("voices", {})

        voices = {}
        for vid, vcfg in voices_raw.items():
            voices[vid] = VoiceConfig(
                ref_audio=vcfg["ref_audio"],
                ref_text=vcfg["ref_text"],
            )

        return cls(
            model_name=model_cfg.get("name", cls.model_name),
            device=model_cfg.get("device", cls.device),
            dtype=model_cfg.get("dtype", cls.dtype),
            model_local_path=model_cfg.get("local_path"),
            max_slots=worker_cfg.get("max_slots", cls.max_slots),
            emit_every_frames=worker_cfg.get("emit_every_frames", cls.emit_every_frames),
            decode_window_frames=worker_cfg.get("decode_window_frames", cls.decode_window_frames),
            overlap_samples=worker_cfg.get("overlap_samples", cls.overlap_samples),
            max_frames=worker_cfg.get("max_frames", cls.max_frames),
            host=server_cfg.get("host", cls.host),
            port=server_cfg.get("port", cls.port),
            voices=voices,
        )

    def to_yaml(self, path: str):
        """Save configuration to a YAML file."""
        data = {
            "model": {
                "name": self.model_name,
                "device": self.device,
                "dtype": self.dtype,
            },
            "worker": {
                "max_slots": self.max_slots,
                "emit_every_frames": self.emit_every_frames,
                "decode_window_frames": self.decode_window_frames,
                "overlap_samples": self.overlap_samples,
                "max_frames": self.max_frames,
            },
            "server": {
                "host": self.host,
                "port": self.port,
            },
        }
        if self.model_local_path:
            data["model"]["local_path"] = self.model_local_path
        if self.voices:
            data["voices"] = {
                vid: {"ref_audio": v.ref_audio, "ref_text": v.ref_text}
                for vid, v in self.voices.items()
            }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
