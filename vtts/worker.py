# vTTS - Continuous Batching for Text-to-Speech
# https://github.com/caimari/vtts
# Copyright (c) 2025 Antoni Caimari Caldes
# Licensed under MIT License

"""
Continuous Generation Worker (M2): slot-based TTS with dynamic join/leave
and batched forward passes.

Each active TTS request occupies a "slot" with its own KV cache and state.
New requests can join mid-generation (dynamic join). Finished slots are
removed immediately (dynamic leave). All active slots share a single
batched forward pass through the transformer each step.

This is the TTS equivalent of what vLLM does for LLM inference.

Architecture:
  - SlotState: per-request state (KV cache, generation_step, codes, etc.)
  - ContinuousWorker: manages slots, runs the batched autoregressive loop
  - Each step: one forward pass for ALL active slots, emit audio when ready
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict

import numpy as np
import torch

logger = logging.getLogger("vtts.worker")


@dataclass
class TTSRequest:
    """A single TTS generation request."""
    request_id: str
    text: str
    voice_id: str
    language: str = "English"
    priority: int = 0
    chunk_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    created_at: float = field(default_factory=time.time)
    first_chunk_at: Optional[float] = None
    finished_at: Optional[float] = None
    total_chunks: int = 0
    total_samples: int = 0
    sample_rate: int = 0


@dataclass
class SlotState:
    """Per-slot state for one active TTS generation."""
    slot_id: int
    request: TTSRequest
    key_caches: list
    value_caches: list
    past_hidden: torch.Tensor
    trailing_text_hidden: torch.Tensor
    rope_delta: torch.Tensor
    generation_step: int
    last_token: torch.Tensor
    codes_buffer: list
    decoded_tail: Optional[np.ndarray] = None
    total_frames_emitted: int = 0
    frames_since_emit: int = 0
    finished: bool = False
    ref_code_context: Optional[torch.Tensor] = None
    ref_code_frames: int = 0
    prefill_seq_len: int = 0
    decode_steps: int = 0


class ContinuousWorker:
    """Continuous batching TTS worker with dynamic join/leave.

    Supports two modes:
      - Built-in speakers (CustomVoice models): use prefill_slot_speaker()
      - Voice cloning (Base models): use prefill_slot() with a VoiceRegistry

    Args:
        model_name: HuggingFace model ID or local path.
        device: CUDA device string.
        dtype: Model precision.
        max_slots: Maximum concurrent generations.
        emit_every_frames: Frames between audio emissions (lower = lower latency).
        decode_window_frames: Vocoder context window.
        overlap_samples: Crossfade overlap between chunks.
        max_frames: Maximum generation length (safety limit).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device: str = "cuda:0",
        dtype=torch.bfloat16,
        max_slots: int = 16,
        emit_every_frames: int = 6,
        decode_window_frames: int = 80,
        overlap_samples: int = 1024,
        max_frames: int = 3600,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.max_slots = max_slots
        self.emit_every_frames = emit_every_frames
        self.decode_window_frames = decode_window_frames
        self.overlap_samples = overlap_samples
        self.max_frames = max_frames

        self.model = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self.slots: Dict[int, SlotState] = {}
        self._next_slot_id = 0

        self._eos_ids = set()
        self._vocab_size = 0
        self._suppress_tokens = []

    def load_model(self):
        """Load the TTS model and warm up."""
        from qwen_tts import Qwen3TTSModel

        logger.info(f"Loading {self.model_name} on {self.device}...")
        t0 = time.time()

        self.model = Qwen3TTSModel.from_pretrained(
            self.model_name,
            device_map=self.device,
            dtype=self.dtype,
        )

        talker_config = self.model.model.config.talker_config
        model_config = self.model.model.config
        self._eos_ids = {
            talker_config.codec_eos_token_id,
            2150, 2157, 151670,
            model_config.tts_eos_token_id,
            model_config.im_end_token_id,
            151643,
        }
        self._vocab_size = talker_config.vocab_size
        self._suppress_tokens = [
            i for i in range(self._vocab_size - 1024, self._vocab_size)
            if i not in self._eos_ids
        ]

        self.model.model.talker.enable_fast_codebook_gen(True)

        # Warmup
        supported = self.model.get_supported_speakers()
        if supported:
            self.model.generate_custom_voice(
                text="hello", speaker=supported[0], language="English",
            )
        logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the asyncio event loop for thread-safe chunk delivery."""
        self._loop = loop

    @property
    def num_active_slots(self) -> int:
        return len(self.slots)

    @property
    def has_capacity(self) -> bool:
        return len(self.slots) < self.max_slots

    def get_supported_speakers(self):
        """Return list of built-in speaker names, or empty list."""
        if self.model is None:
            return []
        return self.model.get_supported_speakers() or []

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prefill_slot_speaker(self, request: TTSRequest) -> SlotState:
        """Prefill a slot using a built-in speaker (CustomVoice models)."""
        model_inner = self.model.model
        model_api = self.model

        input_text = model_api._build_assistant_text(request.text)
        input_ids = model_api._tokenize_texts([input_text])

        talker_input_embeds, talker_attention_mask, trailing_text_hiddens, tts_pad_embed = \
            model_inner._build_talker_inputs(
                input_ids=input_ids,
                instruct_ids=None,
                ref_ids=None,
                voice_clone_prompt=None,
                languages=[request.language],
                speakers=[request.voice_id],
                non_streaming_mode=False,
            )

        return self._run_prefill(request, talker_input_embeds, talker_attention_mask,
                                 trailing_text_hiddens, tts_pad_embed)

    @torch.no_grad()
    def prefill_slot(self, request: TTSRequest, voice_registry) -> SlotState:
        """Prefill a slot using voice cloning (Base models)."""
        voice = voice_registry.get(request.voice_id)
        prompt_item = voice.prompt_item

        model_inner = self.model.model
        model_api = self.model

        input_text = model_api._build_assistant_text(request.text)
        input_ids = model_api._tokenize_texts([input_text])

        voice_clone_prompt = model_api._prompt_items_to_voice_clone_prompt([prompt_item])
        for key in ["ref_code", "ref_spk_embedding"]:
            if voice_clone_prompt.get(key):
                voice_clone_prompt[key] = [
                    t.clone() if t is not None else None
                    for t in voice_clone_prompt[key]
                ]

        ref_ids = None
        if prompt_item.ref_text:
            ref_text_formatted = model_api._build_ref_text(prompt_item.ref_text)
            ref_ids = model_api._tokenize_texts([ref_text_formatted])

        talker_input_embeds, talker_attention_mask, trailing_text_hiddens, tts_pad_embed = \
            model_inner._build_talker_inputs(
                input_ids=input_ids,
                instruct_ids=None,
                ref_ids=ref_ids,
                voice_clone_prompt=voice_clone_prompt,
                languages=[request.language],
                speakers=None,
                non_streaming_mode=False,
            )

        # Extract ref_code context for vocoder
        ref_code_context = None
        ref_code_frames = 0
        if voice_clone_prompt.get("ref_code") and voice_clone_prompt["icl_mode"][0]:
            ref_code = voice_clone_prompt["ref_code"][0]
            if ref_code is not None:
                ref_code_context = ref_code.to(self.device)
                ref_code_frames = ref_code_context.shape[0]

        slot = self._run_prefill(request, talker_input_embeds, talker_attention_mask,
                                 trailing_text_hiddens, tts_pad_embed,
                                 ref_code_context=ref_code_context,
                                 ref_code_frames=ref_code_frames)
        return slot

    def _run_prefill(self, request, talker_input_embeds, talker_attention_mask,
                     trailing_text_hiddens, tts_pad_embed,
                     ref_code_context=None, ref_code_frames=0):
        """Shared prefill logic: forward pass + extract slot state."""
        model_inner = self.model.model

        out = model_inner.talker.forward(
            inputs_embeds=talker_input_embeds,
            attention_mask=talker_attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            trailing_text_hidden=trailing_text_hiddens,
            tts_pad_embed=tts_pad_embed,
            generation_step=None,
            past_hidden=None,
            past_key_values=None,
        )

        from qwen_tts.core.models.modeling_qwen3_tts import _sample_next_token
        token = _sample_next_token(out.logits[:, -1, :], 0.9, 50, 1.0, self._suppress_tokens)

        key_caches = []
        value_caches = []
        for layer_idx in range(len(out.past_key_values)):
            k, v = out.past_key_values[layer_idx]
            key_caches.append(k.clone())
            value_caches.append(v.clone())

        rope_delta = model_inner.talker.rope_deltas.clone() \
            if model_inner.talker.rope_deltas is not None \
            else torch.zeros(1, device=self.device)

        slot_id = self._next_slot_id
        self._next_slot_id += 1

        gen_step = out.generation_step
        slot = SlotState(
            slot_id=slot_id,
            request=request,
            key_caches=key_caches,
            value_caches=value_caches,
            past_hidden=out.past_hidden.clone(),
            trailing_text_hidden=trailing_text_hiddens.clone(),
            rope_delta=rope_delta,
            generation_step=gen_step.item() if isinstance(gen_step, torch.Tensor) else gen_step,
            last_token=token,
            codes_buffer=[],
            ref_code_context=ref_code_context,
            ref_code_frames=ref_code_frames,
            prefill_seq_len=key_caches[0].shape[2],
        )

        self.slots[slot_id] = slot
        logger.info(f"Slot {slot_id} prefilled for {request.request_id} "
                     f"(seq_len={slot.prefill_seq_len})")
        return slot

    # ------------------------------------------------------------------
    # Batched decode step
    # ------------------------------------------------------------------

    def _get_tts_pad_embed(self):
        if not hasattr(self, '_tts_pad_embed'):
            model_inner = self.model.model
            self._tts_pad_embed = model_inner.talker.text_projection(
                model_inner.talker.get_text_embeddings()(
                    torch.tensor(
                        [[model_inner.config.tts_pad_token_id]],
                        device=self.device, dtype=torch.long,
                    )
                )
            )
        return self._tts_pad_embed

    @torch.no_grad()
    def step(self) -> bool:
        """Run a single BATCHED autoregressive step for all active slots.

        All slots are processed in one forward pass through the transformer,
        regardless of their generation_step or KV cache length.

        Returns True if there are still active slots, False if all done.
        """
        if not self.slots:
            return False

        model_inner = self.model.model
        talker = model_inner.talker
        tts_pad_embed = self._get_tts_pad_embed()
        from qwen_tts.core.models.modeling_qwen3_tts import _sample_next_token

        slots_list = list(self.slots.values())
        N = len(slots_list)

        # Phase 1: Code predictor (batched)
        batch_past_hidden = torch.cat([s.past_hidden for s in slots_list], dim=0)
        batch_tokens = torch.stack([s.last_token for s in slots_list], dim=0)
        batch_last_id_hidden = talker.get_input_embeddings()(batch_tokens)
        predictor_input = torch.cat([batch_past_hidden, batch_last_id_hidden], dim=1)

        use_fast = getattr(talker, '_use_fast_codebook_gen', False)
        if use_fast:
            torch.compiler.cudagraph_mark_step_begin()
            codebook_tokens = talker.code_predictor.generate_fast(
                inputs_embeds=predictor_input,
                num_codebooks=talker.config.num_code_groups - 1,
            )
            batch_codec_ids = torch.cat([batch_tokens, codebook_tokens], dim=-1)
            codec_hiddens = torch.cat(
                [batch_last_id_hidden]
                + [talker.code_predictor.get_input_embeddings()[i](codebook_tokens[..., i:i+1])
                   for i in range(talker.config.num_code_groups - 1)],
                dim=1,
            )
        else:
            torch.compiler.cudagraph_mark_step_begin()
            predictor_result = talker.code_predictor.generate(
                inputs_embeds=predictor_input,
                max_new_tokens=talker.config.num_code_groups - 1,
                do_sample=True, top_p=1.0, top_k=50, temperature=1.0,
                output_hidden_states=True, return_dict_in_generate=True,
            )
            batch_codec_ids = torch.cat([batch_tokens, predictor_result.sequences], dim=-1)
            codec_hiddens = torch.cat(
                [batch_last_id_hidden]
                + [talker.code_predictor.get_input_embeddings()[i](predictor_result.sequences[..., i:i+1])
                   for i in range(talker.config.num_code_groups - 1)],
                dim=1,
            )

        batch_inputs_embeds = codec_hiddens.sum(1, keepdim=True)

        # Phase 2: Add trailing_text_hidden per-slot
        for i, slot in enumerate(slots_list):
            if slot.generation_step < slot.trailing_text_hidden.shape[1]:
                batch_inputs_embeds[i] += slot.trailing_text_hidden[0, slot.generation_step].unsqueeze(0)
            else:
                batch_inputs_embeds[i] += tts_pad_embed[0, 0]

        # Phase 3: Pad KV caches, build attention mask
        from transformers.cache_utils import DynamicCache

        seq_lens = [s.key_caches[0].shape[2] for s in slots_list]
        max_seq = max(seq_lens)
        num_layers = len(slots_list[0].key_caches)

        cache = DynamicCache()
        for layer_idx in range(num_layers):
            k0 = slots_list[0].key_caches[layer_idx]
            num_heads, head_dim = k0.shape[1], k0.shape[3]
            batch_k = torch.zeros(N, num_heads, max_seq, head_dim,
                                  dtype=k0.dtype, device=k0.device)
            batch_v = torch.zeros(N, num_heads, max_seq, head_dim,
                                  dtype=k0.dtype, device=k0.device)
            for i, slot in enumerate(slots_list):
                sl = seq_lens[i]
                batch_k[i, :, max_seq - sl:, :] = slot.key_caches[layer_idx][0]
                batch_v[i, :, max_seq - sl:, :] = slot.value_caches[layer_idx][0]
            cache.update(batch_k, batch_v, layer_idx)

        attention_mask = torch.zeros(N, max_seq + 1, dtype=torch.long, device=self.device)
        for i, sl in enumerate(seq_lens):
            attention_mask[i, max_seq - sl:] = 1

        # Phase 4: Position IDs per-slot
        batch_rope_deltas = torch.stack([s.rope_delta.view(1) for s in slots_list], dim=0)
        cache_positions = torch.tensor(seq_lens, device=self.device, dtype=torch.long)
        position_ids = (cache_positions.unsqueeze(1) + batch_rope_deltas).long()
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        talker.rope_deltas = batch_rope_deltas

        # Phase 5: Single batched forward
        outputs = talker.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            inputs_embeds=batch_inputs_embeds,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        hidden_states = outputs.last_hidden_state
        logits = talker.codec_head(hidden_states)

        # Phase 6: Post-process per-slot
        finished_slots = []
        updated_cache = outputs.past_key_values

        for i, slot in enumerate(slots_list):
            codec_ids_i = batch_codec_ids[i]

            if codec_ids_i[0].item() in self._eos_ids:
                slot.finished = True
                finished_slots.append(slot.slot_id)
                new_sl = seq_lens[i] + 1
                for layer_idx in range(num_layers):
                    k, v = updated_cache[layer_idx]
                    slot.key_caches[layer_idx] = k[i:i+1, :, max_seq + 1 - new_sl:, :]
                    slot.value_caches[layer_idx] = v[i:i+1, :, max_seq + 1 - new_sl:, :]
                continue

            token = _sample_next_token(logits[i, -1, :].unsqueeze(0), 0.9, 50, 1.0,
                                       self._suppress_tokens)
            slot.last_token = token
            slot.past_hidden = hidden_states[i:i+1]
            slot.generation_step += 1

            new_sl = seq_lens[i] + 1
            for layer_idx in range(num_layers):
                k, v = updated_cache[layer_idx]
                slot.key_caches[layer_idx] = k[i:i+1, :, max_seq + 1 - new_sl:, :].clone()
                slot.value_caches[layer_idx] = v[i:i+1, :, max_seq + 1 - new_sl:, :].clone()

            slot.codes_buffer.append(codec_ids_i.detach())
            slot.frames_since_emit += 1
            slot.decode_steps += 1

        for sid in finished_slots:
            slot = self.slots[sid]
            self._flush_slot(slot)
            self._put_chunk(slot.request.chunk_queue, None)
            slot.request.finished_at = time.time()
            del self.slots[sid]
            logger.info(f"Slot {sid} finished ({slot.request.request_id})")

        self._try_emit_chunks()

        for slot in list(self.slots.values()):
            if slot.decode_steps >= self.max_frames:
                slot.finished = True
                self._flush_slot(slot)
                self._put_chunk(slot.request.chunk_queue, None)
                slot.request.finished_at = time.time()
                del self.slots[slot.slot_id]

        return bool(self.slots)

    # ------------------------------------------------------------------
    # Audio emission
    # ------------------------------------------------------------------

    def _try_emit_chunks(self):
        """Decode and emit audio chunks for slots that have enough frames."""
        from qwen_tts.core.models.modeling_qwen3_tts import _add_ref_code_context, _crossfade

        model_inner = self.model.model
        samples_per_frame = model_inner.speech_tokenizer.get_decode_upsample_rate()
        step_samples = samples_per_frame * self.emit_every_frames

        ready_slots = []
        ready_windows = []

        for slot in self.slots.values():
            if slot.finished or slot.frames_since_emit < self.emit_every_frames:
                continue
            if not slot.codes_buffer:
                continue

            start = max(0, len(slot.codes_buffer) - self.decode_window_frames)
            window_codes = torch.stack(slot.codes_buffer[start:], dim=0)
            window, _ = _add_ref_code_context(
                window_codes, slot.ref_code_context, slot.ref_code_frames,
                self.decode_window_frames,
            )
            ready_slots.append(slot)
            ready_windows.append(window)

        if not ready_windows:
            return

        max_t = max(w.shape[0] for w in ready_windows)
        padded_windows = []
        for w in ready_windows:
            if w.shape[0] < max_t:
                pad = torch.zeros(max_t - w.shape[0], w.shape[1],
                                  dtype=w.dtype, device=w.device)
                padded_windows.append(torch.cat([pad, w], dim=0))
            else:
                padded_windows.append(w)

        batch_codes = torch.stack(padded_windows, dim=0).to(self.device)

        batch_wavs = []
        for i in range(batch_codes.shape[0]):
            wavs_i, sr = model_inner.speech_tokenizer.decode(
                [{"audio_codes": batch_codes[i]}]
            )
            batch_wavs.append(wavs_i[0])

        from qwen_tts.core.models.modeling_qwen3_tts import _crossfade

        for idx, slot in enumerate(ready_slots):
            wav = batch_wavs[idx].astype(np.float32)
            chunk = wav[-step_samples:] if step_samples > 0 and len(wav) > step_samples else wav

            if slot.decoded_tail is not None and self.overlap_samples > 0:
                ov = min(self.overlap_samples, len(slot.decoded_tail), len(chunk))
                if ov > 0:
                    head = _crossfade(slot.decoded_tail[-ov:], chunk[:ov])
                    chunk = np.concatenate([head, chunk[ov:]], axis=0)

            if slot.decoded_tail is None:
                fade_len = min(self.overlap_samples, len(chunk))
                if fade_len > 0:
                    t = np.arange(fade_len, dtype=np.float32) / max(fade_len - 1, 1)
                    fade_in = 0.5 * (1 - np.cos(np.pi * t))
                    chunk[:fade_len] *= fade_in

            slot.decoded_tail = wav.copy()
            slot.total_frames_emitted = len(slot.codes_buffer)
            slot.frames_since_emit = 0

            if self.overlap_samples > 0 and len(chunk) > self.overlap_samples * 2:
                chunk = chunk[:-self.overlap_samples]

            slot.request.sample_rate = sr
            if slot.request.first_chunk_at is None:
                slot.request.first_chunk_at = time.time()
            slot.request.total_chunks += 1
            slot.request.total_samples += len(chunk)
            self._put_chunk(slot.request.chunk_queue, chunk)

    def _flush_slot(self, slot: SlotState):
        """Decode and emit remaining frames for a finished slot."""
        if not slot.codes_buffer:
            return
        remaining_frames = len(slot.codes_buffer) - slot.total_frames_emitted
        if remaining_frames <= 0:
            return

        model_inner = self.model.model
        context_frames = min(slot.total_frames_emitted,
                             self.decode_window_frames - remaining_frames)
        start_idx = max(0, slot.total_frames_emitted - context_frames)
        window_codes = torch.stack(slot.codes_buffer[start_idx:], dim=0)

        from qwen_tts.core.models.modeling_qwen3_tts import _add_ref_code_context, _crossfade
        window, ref_prefix = _add_ref_code_context(
            window_codes, slot.ref_code_context, slot.ref_code_frames,
            self.decode_window_frames,
        )

        wavs, sr = model_inner.speech_tokenizer.decode([{"audio_codes": window}])
        wav = wavs[0].astype(np.float32)

        samples_per_frame = model_inner.speech_tokenizer.get_decode_upsample_rate()
        skip_samples = (ref_prefix + context_frames) * samples_per_frame
        if 0 < skip_samples < len(wav):
            wav = wav[skip_samples:]

        if slot.decoded_tail is not None and self.overlap_samples > 0:
            ov = min(self.overlap_samples, len(slot.decoded_tail), len(wav))
            if ov > 0:
                head = _crossfade(slot.decoded_tail[-ov:], wav[:ov])
                wav = np.concatenate([head, wav[ov:]], axis=0)

        fade_len = min(self.overlap_samples, len(wav))
        if fade_len > 0:
            t = np.arange(fade_len, dtype=np.float32) / max(fade_len - 1, 1)
            fade_out = 0.5 * (1 + np.cos(np.pi * t))
            wav[-fade_len:] *= fade_out

        slot.request.sample_rate = sr
        if slot.request.first_chunk_at is None:
            slot.request.first_chunk_at = time.time()
        slot.request.total_chunks += 1
        slot.request.total_samples += len(wav)
        self._put_chunk(slot.request.chunk_queue, wav)

    def _put_chunk(self, queue: asyncio.Queue, chunk):
        """Thread-safe put to asyncio queue."""
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(queue.put_nowait, chunk)
        else:
            queue.put_nowait(chunk)
