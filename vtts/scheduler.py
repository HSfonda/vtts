# vTTS - Continuous Batching for Text-to-Speech
# https://github.com/caimari/vtts
# Copyright (c) 2025 Antoni Caimari Caldes
# Licensed under MIT License

"""
Continuous Scheduler: async loop that feeds the ContinuousWorker.

Alternates between:
  1. Draining pending requests (prefill + add to slots)
  2. Running worker.step() for all active slots
"""

import asyncio
import logging
import time
from typing import Optional

from vtts.worker import ContinuousWorker, TTSRequest

logger = logging.getLogger("vtts.scheduler")


class ContinuousScheduler:
    """Async scheduler for continuous batching TTS.

    Usage:
        scheduler = ContinuousScheduler(worker, voice_registry)
        asyncio.create_task(scheduler.run())
        await scheduler.submit(request)
    """

    def __init__(
        self,
        worker: ContinuousWorker,
        voice_registry=None,
        idle_sleep: float = 0.001,
    ):
        self.worker = worker
        self.voice_registry = voice_registry
        self.idle_sleep = idle_sleep
        self._pending: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._built_in_speakers = set()

    async def start(self):
        """Start the scheduler loop."""
        supported = self.worker.get_supported_speakers()
        self._built_in_speakers = {s.lower() for s in supported} if supported else set()
        self._running = True
        logger.info("Scheduler started")
        await self._loop()

    async def stop(self):
        self._running = False

    async def submit(self, request: TTSRequest):
        """Submit a TTS request for generation."""
        await self._pending.put(request)

    async def _loop(self):
        while self._running:
            # Drain pending requests
            while not self._pending.empty() and self.worker.has_capacity:
                try:
                    request = self._pending.get_nowait()
                except asyncio.QueueEmpty:
                    break

                if request.voice_id.lower() in self._built_in_speakers:
                    self.worker.prefill_slot_speaker(request)
                elif self.voice_registry and request.voice_id in self.voice_registry:
                    self.worker.prefill_slot(request, self.voice_registry)
                else:
                    logger.error(f"Unknown voice '{request.voice_id}' for {request.request_id}")
                    request.chunk_queue.put_nowait(None)
                    continue

            # Step all active slots
            if self.worker.num_active_slots > 0:
                self.worker.step()
            else:
                await asyncio.sleep(self.idle_sleep)
