"""Simulated live acquisition: feed projections to a reconstructor in groups.

The 'sink' is anything with add_projections(indices, block) - a local
SliceReconstructor or a network connection to `minirecast serve`.
"""
import threading
import time

import numpy as np


def arrival_order(n, mode="sequential"):
    """Order in which projections are 'acquired'.

    sequential  : 0, 1, 2, ... (normal continuous rotation - the image fills in
                  as the angular range grows: limited-angle artefacts at first)
    interleaved : bit-reversed order - every group spreads evenly over the full
                  range, so a sharp image appears early and then gets less noisy
    """
    if mode == "sequential":
        return np.arange(n)
    if mode == "interleaved":
        bits = max(1, int(np.ceil(np.log2(n))))
        idx = np.arange(2 ** bits)
        rev = np.zeros_like(idx)
        for b in range(bits):
            rev |= ((idx >> b) & 1) << (bits - 1 - b)
        return rev[rev < n]
    raise ValueError(f"unknown order {mode!r}")


class Feeder(threading.Thread):
    """Push `proj` ([rows, angles, cols]) into `sink` at `rate` projections/s."""

    def __init__(self, sink, proj, order="sequential", group=20, rate=100.0, verbose=True):
        super().__init__(daemon=True)
        self.sink = sink
        self.proj = proj
        self.order = arrival_order(proj.shape[1], order)
        self.group = max(1, int(group))
        self.rate = float(rate)
        self.verbose = verbose
        self.stop_event = threading.Event()
        self.sent = 0

    def run(self):
        t0 = time.perf_counter()
        for i in range(0, len(self.order), self.group):
            if self.stop_event.is_set():
                return
            idx = self.order[i:i + self.group]
            self.sink.add_projections(idx, np.ascontiguousarray(self.proj[:, idx, :]))
            self.sent = i + len(idx)
            if self.rate > 0:                        # keep the acquisition rate
                wait = t0 + self.sent / self.rate - time.perf_counter()
                if wait > 0:
                    self.stop_event.wait(wait)
        if self.verbose:
            dt = time.perf_counter() - t0
            print(f"[stream] {self.sent} projections streamed in {dt:.1f}s "
                  f"({self.sent / max(dt, 1e-9):.0f}/s)")

    def stop(self):
        self.stop_event.set()
