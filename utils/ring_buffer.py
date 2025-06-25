"""
utils.ring_buffer
-----------------
A minimal, NumPy-backed, constant-time rolling buffer.

Example
-------
buf = RingBuffer(capacity=100, item_shape=(), dtype=np.float32)
buf.push(42)
if buf.is_full:
    latest = buf.view()        # newest item is last
"""
from __future__ import annotations
import numpy as np


class RingBuffer:
    """Fixed-capacity circular buffer with O(1) inserts and views."""

    def __init__(self, capacity: int, item_shape: tuple[int, ...] = (),
                 dtype=np.float32):
        self.capacity = int(capacity)
        self._data    = np.empty((capacity, *item_shape), dtype=dtype)
        self._start   = 0          # index of the OLDEST element
        self._size    = 0          # current number of valid elements

    # ------------------------------------------------------------------ #
    # public API                                                         #
    # ------------------------------------------------------------------ #
    def push(self, item: np.ndarray | float | int) -> None:
        """Append *item* (newest) and discard the oldest if buffer is full."""
        idx = (self._start + self._size) % self.capacity
        self._data[idx] = item
        if self._size < self.capacity:
            self._size += 1
        else:                       # overwrite ⇒ window slides fwd by one
            self._start = (self._start + 1) % self.capacity

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity

    def view(self, newest_last: bool = True) -> np.ndarray:
        """Return a *contiguous* view of valid data.

        newest_last=True  ⇒ chronological order (oldest…newest)
        newest_last=False ⇒ reverse order   (newest…oldest)
        """
        if self._size == 0:
            raise RuntimeError("Buffer is empty")
        idx = (self._start + np.arange(self._size)) % self.capacity
        arr = self._data[idx]
        return arr if newest_last else arr[::-1].copy()
