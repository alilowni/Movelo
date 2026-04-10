# pipeline factory — chains named steps, passes a shared context dict.
# steps can be added, inserted, or removed to customize the flow.

from __future__ import annotations

import time
from typing import Callable

# each step takes a context dict and returns it (possibly modified)
StepFn = Callable[[dict], dict]


class Pipeline:

    def __init__(self) -> None:
        self._steps: list[tuple[str, StepFn]] = []

    # step registration

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

    # execution — runs all steps in order, prints timing per step

    def run(self, context: dict | None = None) -> dict:
        ctx = context if context is not None else {}
        total = len(self._steps)
        for i, (name, fn) in enumerate(self._steps, 1):
            t0 = time.time()
            print(f"[{i}/{total}] {name} ", end="", flush=True)
            ctx = fn(ctx)
            elapsed = time.time() - t0
            print(f" ({elapsed:.1f}s)")
        print(f"\nDone — {total} steps complete.")
        return ctx
