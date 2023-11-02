import bisect
import copy
from typing import Iterable, Optional, Tuple

from util.py_util import is_iterable


class _InfiniteSequenceSlice:
    """
    This is a proxy object used to support the += and -= operators for InfiniteSequence.

    See InfiniteSequence.__getitem__().
    """
    def __init__(self, seq: 'InfiniteSequence', start, stop):
        self.seq = seq
        self.start = start
        self.stop = stop
        self.add_delta = None

    def __iadd__(self, delta):
        self.add_delta = delta
        return self

    def __isub__(self, delta):
        self.add_delta = -delta
        return self


class InfiniteSequence:
    """
    Represents an infinite sequence S: S[0], S[1], S[2], ...

    Initially all elements will be equal to the same value, which is specified at construction.

    You are able to access and modify the sequence using the bracket operator, and it will look
    and feel like you are working with a list with setters that have numpy-broadcasting-mechanics.
    For example, you can do:

    S = InfiniteSequence(0)
    S[:5] = 3
    S[10:100] = 1
    S[20:30] += 1
    S[25] = 4
    S[9999999:] += 5

    The nature of the permitted mutators guarantees that after n modifications, the sequence will
    contain at most (n+1) distinct values.

    The underlying representation uses O(n) space, where n is the number of times you modify the
    sequence. Note that the actual indices you modify are not relevant for the space complexity.
    In the above example, the choice of 9999999 does not impact the memory usage.
    """
    def __init__(self, initial_value=0.0, value_fmt='%.3f'):
        """
        Constructs an InfiniteSequence whose elements are all equal to initial_value.

        value_fmt is a format string used to convert values to strings and is used in the
        __str__() method.
        """
        self._indices = [0]
        self._values = [initial_value]
        self._max_values = [initial_value]  # self._max_values[i] == max(self._values[i:])
        self._value_fmt = value_fmt

    def debug_print(self, msg: str):
        print(f'--- {msg} ---')
        print('indices:', self._indices)
        print('values: [%s]' % (', '.join(self._value_fmt % v for v in self._values)))
        print('max_values: [%s]' % (', '.join(self._value_fmt % v for v in self._max_values)))

    def get_start(self, max_value: float) -> Optional[int]:
        """
        Returns the minimum N with the property that S[n] <= max_value for all n >= N.

        If no such N exists, returns None.
        """
        k = bisect.bisect_left(self._max_values, -max_value, key=lambda w: -w)
        if k == len(self._indices):
            return None
        return self._indices[k]

    def check_invariants(self):
        assert len(self._indices) == len(self._values)
        assert len(self._indices) >= 1
        for i in range(len(self._indices) - 1):
            assert self._indices[i] < self._indices[i + 1]
            assert self._values[i] != self._values[i + 1]
            assert self._max_values[i] == max(self._values[i], self._max_values[i + 1])
        assert self._indices[0] == 0

    def to_string(self, delim: str, cap=None) -> str:
        """
        Helper to __str__(). This is separated out in order to give greater control over the
        formatting.

        delim is used to separate the tokens.

        If cap is specified, then all values that are > cap will be replaced with cap.
        """
        indices = self._indices
        values = self._values
        if cap is not None:
            eps = 1e-6  # arbitrary positive constant
            clone = copy.deepcopy(self)
            clone._values = [min(v, cap + eps) for v in values]
            clone._collapse(range(len(indices)))
            # the max_values invariant is broken after this collapse, but we don't need to fix it

            indices = clone._indices
            values = clone._values

        tokens = []
        for i in range(len(indices) - 1):
            start = indices[i]
            end = indices[i + 1]
            value = values[i]
            suffix = ''
            if cap is not None and value > cap:
                # if we got here then eps was added to the value
                value = cap
                suffix = '+'
            tokens.append('[%d, %d): %s%s' % (start, end, self._value_fmt % value, suffix))

        tokens.append('[%d, inf): %s' % (indices[-1], self._value_fmt % values[-1]))
        return delim.join(tokens)

    def __str__(self):
        return 'WeightSequence(%s)' % self.to_string(delim=', ')

    def __getitem__(self, index):
        """
        If index is an integer, returns S[index].

        If index is a slice, then returns an _InfiniteSequenceSlice object. This is a proxy object
        used to support the += operator. Slices are only to be used for setting, not for getting,
        although there isn't a way to enforce this.
        """
        if isinstance(index, slice):
            return _InfiniteSequenceSlice(self, index.start, index.stop)

        assert isinstance(index, int), index
        assert index >= self._indices[0], (index, self._indices[0])
        k = bisect.bisect_right(self._indices, index)
        return self._values[k - 1]

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            assert index.start is None or index.start >= 0
            assert index.stop is None or index.stop >= 0
            assert None in (index.start, index.stop) or index.start <= index.stop
            if isinstance(value, _InfiniteSequenceSlice):
                assert value.add_delta is not None
                self._increment(index.start, index.stop, value.add_delta)
            else:
                self._set(index.start, index.stop, value)
            return

        assert isinstance(index, int), index
        assert index >= self._indices[0], (index, self._indices[0])
        self._set(index, index + 1, value)

    def _collapse(self, indices_to_check: Iterable[int]):
        """
        After some modifications, it is possible that the sequence contains consecutive repeated
        values, which would break the invariant. This function fixes this by merging consecutive
        repeated values.

        This will only check for

        self._values[i-1] == self._values[i]

        for i in indices_to_check. The caller must take care to ensure that indices_to_check is
        specified correctly to cover the possible invariant breaks.
        """
        n = len(self._indices)

        for i in sorted(set(indices_to_check), reverse=True):
            if i <= 0 or i >= n:
                continue
            if self._values[i-1] == self._values[i]:
                self._indices.pop(i)
                self._values.pop(i)
                self._max_values.pop(i)

    def _insert_indices(self, start: Optional[int], end: Optional[int]) -> Tuple[int, int]:
        """
        Helper to _set() and _increment().

        Inserts start and end into self._indices if either is not currently present in
        self._indices. Upon such insertions, makes corresponding insertions to self._values and
        self._max_values, copying the previous value in those lists.

        If start is None, then it is interpreted as 0. If end is None, then it is ignored.

        Note that this can break the invariant that self._values does not contain repeated values.
        This invariant must be restored later by calling self._collapse().

        Returns the two values (a, b) such that

        (self._indices[a], self._indices[b]) == (start, end)

        If start is None, then a is 0. If end is None, then b is len(self._indices)
        """
        start = 0 if start is None else start
        a = bisect.bisect_left(self._indices, start)
        if a == len(self._indices) or self._indices[a] != start:
            assert a > 0
            self._indices.insert(a, start)
            self._values.insert(a, self._values[a - 1])
            self._max_values.insert(a, self._max_values[a - 1])

        if end is not None:
            b = bisect.bisect_left(self._indices, end)
            if b == len(self._indices) or self._indices[b] != end:
                assert b > 0
                self._indices.insert(b, end)
                self._values.insert(b, self._values[b - 1])
                self._max_values.insert(b, self._max_values[b - 1])
        else:
            b = len(self._indices)

        return (a, b)

    def _recalculate_max_values(self, end: int):
        """
        Recomputes self._max_values for indices < end.
        """
        n = len(self._max_values)
        i = end - 1
        assert i < n

        if i == n - 1:
            self._max_values[i] = self._values[i]
            i -= 1

        while i >= 0:
            self._max_values[i] = max(self._values[i], self._max_values[i + 1])
            i -= 1

    def _set(self, start: Optional[int], end: Optional[int], value):
        """
        Performs S[start:end] = value

        Helper to __setitem__().
        """
        assert not is_iterable(value), value
        a, b = self._insert_indices(start, end)

        self._indices = self._indices[:a+1] + self._indices[b:]
        self._values = self._values[:a] + [value] + self._values[b:]
        self._max_values = self._max_values[:a] + [value] + self._max_values[b:]

        self._recalculate_max_values(a + 2)
        self._collapse((a, a + 1))

    def _increment(self, start: Optional[int], end: Optional[int], delta):
        """
        Performs S[start:end] += delta

        Helper to __setitem__().
        """
        assert not is_iterable(delta), delta
        a, b = self._insert_indices(start, end)
        n = len(self._max_values)

        i = b - 1
        assert i < n

        if i + 1 == n:
            self._values[i] += delta
            self._max_values[i] = self._values[i]
            i -= 1

        while i >= a:
            self._values[i] += delta
            self._max_values[i] = max(self._values[i], self._max_values[i + 1])
            i -= 1

        self._recalculate_max_values(b)
        self._collapse((a, b))
