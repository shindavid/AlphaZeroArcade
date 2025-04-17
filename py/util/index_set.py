import numpy as np
from typing import Optional

class IndexSet:
    """
    A set-like data structure for storing non-negative integers using a NumPy boolean array.

    This class behaves like a set of integers (i.e., supports membership testing, adding/removing elements,
    iteration, and length), but is implemented using a dynamic NumPy bit array for performance and memory efficiency.

    ### Key Features
    - Internally stores membership using a NumPy array of dtype `bool`
    - Automatically resizes to accommodate new indices
    - Supports NumPy-style inversion: `invert(n)` returns a new `IndexSet` containing indices in `[0, n)` not present in the original
    - Iterable and convertible to a NumPy array (`np.array(index_set)`)

    ### Example:
    >>> s = IndexSet()
    >>> s.add(3)
    >>> s.add(7)
    >>> 3 in s
    True
    >>> list(s)
    [3, 7]
    >>> s.remove(3)
    >>> print(s)
    IndexSet([7])
    >>> inv = s.invert(10)
    >>> print(inv)
    IndexSet([0, 1, 2, 3, 4, 5, 6, 8, 9])
    """

    def __init__(self):
        self.bits = np.zeros(8, dtype=bool)

    @classmethod
    def from_bits(cls, bits: np.ndarray):
        obj = cls()
        obj.bits = bits.astype(bool)
        return obj

    def _ensure_capacity(self, index: int):
        if index >= self.bits.size:
            new_size = max(index + 1, self.bits.size * 2)
            new_bits = np.zeros(new_size, dtype=bool)
            new_bits[:self.bits.size] = self.bits
            self.bits = new_bits

    def add(self, value: int):
        if value < 0:
            raise IndexError(f"Negative index {value} is not allowed")
        self._ensure_capacity(value)
        self.bits[value] = True

    def remove(self, value: int):
        if value < 0 or value >= self.bits.size or not self.bits[value]:
            raise KeyError(value)
        self.bits[value] = False

    def discard(self, value: int):
        if 0 <= value < self.bits.size:
            self.bits[value] = False

    def invert(self, n: Optional[int]=None) -> 'IndexSet':
        """
        Returns an IndexSet that contains all integers i in the range [0, n) where i not in self.
        """
        if n is None:
            n = self.bits.size

        if n <= 0:
            return IndexSet()

        bits = np.ones(n, dtype=bool)
        if len(self.bits) == 0:
            return IndexSet.from_bits(bits)

        k = min(n, self.bits.size)
        bits[:k] = ~self.bits[:k]
        return IndexSet.from_bits(bits)

    def __contains__(self, value: int) -> bool:
        return 0 <= value < self.bits.size and self.bits[value]

    def __iter__(self):
        return iter(np.where(self.bits)[0])

    def __len__(self):
        return int(np.count_nonzero(self.bits))

    def __repr__(self):
        return f"IndexSet({list(self)})"

    def __array__(self, dtype=None, copy=True):
        return self.bits

    def __getitem__(self, index):
        return self.bits[index]