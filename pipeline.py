"""
Pipeline factory for the movelo marketing workflow.

Provides a simple, extensible Pipeline that chains named steps.
Each step is a callable: fn(context: dict) -> dict.
"""

from __future__ import annotations

import time
from typing import Callable

StepFn = Callable[[dict], dict]


class Pipeline:
    """Ordered sequence of named steps sharing a mutable context dict."""

    def __init__(self) -> None:
        self._steps: list[tuple[str, StepFn]] = []

    def register_step(self, name: str, fn: StepFn) -> "Pipeline":
        self._steps.append((name, fn))
        return self

    def insert_before(self, ref_name: str, name: str, fn: StepFn) -> "Pipeline":
        for i, (n, _) in enumerate(self._steps):
            if n == ref_name:
                self._steps.insert(i, (name, fn))
                return self
        raise KeyError(f"Step '{ref_name}' not found in pipeline")

    def insert_after(self, ref_name: str, name: str, fn: StepFn) -> "Pipeline":
        for i, (n, _) in enumerate(self._steps):
            if n == ref_name:
                self._steps.insert(i + 1, (name, fn))
                return self
        raise KeyError(f"Step '{ref_name}' not found in pipeline")

    def remove_step(self, name: str) -> "Pipeline":
        self._steps = [(n, fn) for n, fn in self._steps if n != name]
        return self

    @property
    def step_names(self) -> list[str]:
        return [n for n, _ in self._steps]

    def run(self, context: dict | None = None) -> dict:
        ctx = context if context is not None else {}
        total = len(self._steps)
        for i, (name, fn) in enumerate(self._steps, 1):
            t0 = time.time()
            print(f"\n[{i}/{total}] {name}")
            ctx = fn(ctx)
            elapsed = time.time() - t0
            print(f"  done ({elapsed:.1f}s)")
        print(f"\nPipeline complete ({total} steps)")
        return ctx
