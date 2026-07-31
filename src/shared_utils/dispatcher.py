"""Device dispatcher for the parallel processing pipeline.

The dispatcher sits between the pipeline's worker threads and the set of
available compute devices (CPU + N GPUs). Its job is to keep GPUs busy
when they have headroom, and let flexible processors fall back to CPU
when they don't — without blocking and without re-building processors.

Design constraints (see ``DISPATCHER_FEASIBILITY_REPORT.md`` at the repo
root for the full rationale):

  * **Non-blocking acquire.** A flexible worker never waits for a GPU
    slot; if every GPU's semaphore is saturated, ``acquire`` returns a
    CPU lease immediately. The whole point of having flexible steps is
    that CPU is a legitimate execution target.
  * **Pinned GPU steps are not managed here.** ``cloud_mask`` and
    ``cloud_height_emulator`` hold their GPU for the life of the run;
    the dispatcher only allocates *flexible* leases against a capped
    per-GPU budget that the operator sizes to leave the pinned workload
    room.
  * **No preemption.** Leases are short (one scene at one step) and the
    dispatcher does not revoke them. Adaptation to changing load happens
    at the grain of "the next scene".
"""
from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Lease primitive
# --------------------------------------------------------------------- #

class DeviceLease:
    """A short-term right to run on a specific compute device.

    Used as a context manager by the worker thread:

        with dispatcher.acquire(step_name) as lease:
            processor = proc_cache[lease.device]
            result = processor.process(scene, ...)

    On ``__exit__`` the underlying semaphore slot (if any) is released.
    CPU leases hold no semaphore and are effectively free — but still go
    through the same interface so the worker doesn't branch on device.
    """

    __slots__ = ("device", "_sem", "_dispatcher", "_step")

    def __init__(
        self,
        device: str,
        semaphore: Optional[threading.Semaphore],
        dispatcher: "DeviceDispatcher",
        step: str,
    ) -> None:
        self.device = device
        self._sem = semaphore
        self._dispatcher = dispatcher
        self._step = step

    def __enter__(self) -> "DeviceLease":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._sem is not None:
            self._sem.release()

    def __repr__(self) -> str:  # pragma: no cover — debug only
        return f"DeviceLease(step={self._step!r}, device={self.device!r})"


# --------------------------------------------------------------------- #
# Stats — lightweight lease accounting for logging
# --------------------------------------------------------------------- #

@dataclass
class _LeaseStats:
    gpu: int = 0
    cpu: int = 0

    @property
    def total(self) -> int:
        return self.gpu + self.cpu

    @property
    def cpu_fraction(self) -> float:
        return self.cpu / self.total if self.total else 0.0


# --------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------- #

class DeviceDispatcher:
    """Hands out short-term device leases to flexible-step workers.

    Args:
        n_gpus: Number of visible CUDA devices. 0 ⇒ every lease returns
            CPU (dispatcher is effectively a no-op).
        flex_slots_per_gpu: Maximum concurrent *flexible* leases allowed
            per GPU. Tuned downward when pinned steps eat most of VRAM.
            A value of 1 means flex steps only use a GPU when no other
            flex lease is already running there.
    """

    def __init__(self, n_gpus: int, flex_slots_per_gpu: int = 1) -> None:
        self.n_gpus = max(0, int(n_gpus))
        self.flex_slots_per_gpu = max(1, int(flex_slots_per_gpu))

        # Per-GPU semaphore. Using threading.BoundedSemaphore so a buggy
        # double-release is caught immediately rather than growing the cap.
        self._gpu_semaphores: Dict[str, threading.BoundedSemaphore] = {
            f"cuda:{i}": threading.BoundedSemaphore(self.flex_slots_per_gpu)
            for i in range(self.n_gpus)
        }

        # Round-robin cursor across GPUs. Protected by _rr_lock.
        self._rr_cursor = 0
        self._rr_lock = threading.Lock()

        # Stats: step-name → _LeaseStats. Lock covers all reads/writes.
        self._stats: Dict[str, _LeaseStats] = defaultdict(_LeaseStats)
        self._stats_lock = threading.Lock()

    # -- public API ---------------------------------------------------- #

    def acquire(self, step: str) -> DeviceLease:
        """Return a lease for *step*.

        Tries each GPU in round-robin order; falls back to CPU the
        instant every GPU is saturated. Never blocks.
        """
        if self.n_gpus == 0:
            return self._cpu_lease(step)

        gpus = list(self._gpu_semaphores.keys())
        # Snapshot + bump the cursor atomically. The cursor is a hint
        # only — correctness doesn't depend on it, but concurrent bumps
        # should not step on each other.
        with self._rr_lock:
            start = self._rr_cursor
            self._rr_cursor = (start + 1) % self.n_gpus

        for offset in range(self.n_gpus):
            dev = gpus[(start + offset) % self.n_gpus]
            sem = self._gpu_semaphores[dev]
            if sem.acquire(blocking=False):
                self._record(step, "gpu")
                return DeviceLease(dev, sem, self, step)

        return self._cpu_lease(step)

    def describe(self) -> str:
        """One-line summary suitable for a startup banner."""
        if self.n_gpus == 0:
            return "DeviceDispatcher(cpu-only)"
        return (
            f"DeviceDispatcher(gpus={self.n_gpus}, "
            f"flex_slots_per_gpu={self.flex_slots_per_gpu})"
        )

    def stats_summary(self) -> str:
        """Multi-line lease-counter summary for end-of-run logging."""
        with self._stats_lock:
            if not self._stats:
                return "DeviceDispatcher: no flex leases issued."
            rows = []
            for step, s in sorted(self._stats.items()):
                rows.append(
                    f"  {step:>18}: gpu={s.gpu}  cpu={s.cpu}  "
                    f"({s.cpu_fraction:.0%} on CPU)"
                )
        return "DeviceDispatcher lease counts:\n" + "\n".join(rows)

    def snapshot(self) -> Dict[str, Dict[str, int]]:
        """Return a deep copy of the stats — useful for tests and probes."""
        with self._stats_lock:
            return {
                step: {"gpu": s.gpu, "cpu": s.cpu}
                for step, s in self._stats.items()
            }

    # -- internals ----------------------------------------------------- #

    def _cpu_lease(self, step: str) -> DeviceLease:
        self._record(step, "cpu")
        return DeviceLease("cpu", None, self, step)

    def _record(self, step: str, kind: str) -> None:
        with self._stats_lock:
            if kind == "gpu":
                self._stats[step].gpu += 1
            else:
                self._stats[step].cpu += 1


__all__ = ["DeviceDispatcher", "DeviceLease"]
