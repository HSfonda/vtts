# vTTS - Continuous Batching for Text-to-Speech
# https://github.com/caimari/vtts
# Copyright (c) 2025 Antoni Caimari Caldes
# Licensed under MIT License

"""
Voice Registry: pre-computes and caches voice clone prompts.

Speaker embeddings and reference codec codes are computed ONCE per voice
and reused across all subsequent requests — no repeated GPU work.

Only needed for Base models (voice cloning). CustomVoice models use
built-in speaker names directly.
"""

import hashlib
import threading
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class RegisteredVoice:
    """A registered voice with pre-computed prompt data."""
    voice_id: str
    name: str
    ref_audio_path: str
    ref_text: str
    prompt_item: object  # VoiceClonePromptItem


class VoiceRegistry:
    """Thread-safe registry of pre-computed voice clone prompts.

    Usage:
        registry = VoiceRegistry()
        registry.register(
            model=model,
            ref_audio_path="./voices/my_voice.wav",  # ~5-10 seconds
            ref_text="Exact transcript of the audio.",
            voice_id="my_voice",
        )
    """

    def __init__(self):
        self._voices: Dict[str, RegisteredVoice] = {}
        self._lock = threading.Lock()

    def register(
        self,
        model,
        ref_audio_path: str,
        ref_text: str,
        name: Optional[str] = None,
        voice_id: Optional[str] = None,
    ) -> str:
        """Register a voice by pre-computing speaker embedding and ref codes.

        Args:
            model: Loaded Qwen3TTSModel instance.
            ref_audio_path: Path to reference audio WAV file (~5-10 seconds).
            ref_text: Exact transcript of the reference audio.
            name: Human-readable name for the voice.
            voice_id: Optional explicit ID. Auto-generated from path hash if None.

        Returns:
            The voice_id string.
        """
        if voice_id is None:
            h = hashlib.sha256(ref_audio_path.encode()).hexdigest()[:12]
            voice_id = f"voice_{h}"

        if name is None:
            name = voice_id

        prompt_items = model.create_voice_clone_prompt(
            ref_audio=ref_audio_path,
            ref_text=ref_text,
        )
        prompt_item = prompt_items[0]

        voice = RegisteredVoice(
            voice_id=voice_id,
            name=name,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            prompt_item=prompt_item,
        )

        with self._lock:
            self._voices[voice_id] = voice

        return voice_id

    def get(self, voice_id: str) -> RegisteredVoice:
        """Get a registered voice by ID. Raises KeyError if not found."""
        with self._lock:
            return self._voices[voice_id]

    def list_voices(self) -> list:
        """List all registered voices."""
        with self._lock:
            return [
                {"voice_id": v.voice_id, "name": v.name, "ref_audio": v.ref_audio_path}
                for v in self._voices.values()
            ]

    def __contains__(self, voice_id: str) -> bool:
        with self._lock:
            return voice_id in self._voices

    def __len__(self) -> int:
        with self._lock:
            return len(self._voices)
