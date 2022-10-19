from collections import defaultdict
import time


class Profiler:
    __slots__ = ['_total_sec', '_count', '_start_t', '_min', '_max']

    def __init__(self):
        self._total_sec = 0
        self._count = 0
        self._start_t = 0
        self._min = 9999999999
        self._max = 0

    def start(self, t=None):
        t = time.time() if t is None else t
        self._start_t = t

    def stop(self):
        t = time.time()
        delta = t - self._start_t
        self._total_sec += delta
        self._count += 1
        self._min = min(self._min, delta)
        self._max = max(self._max, delta)
        return t

    def get_total_sec(self):
        return self._total_sec

    def get_count(self):
        return self._count

    def get_avg_sec(self):
        return self._total_sec / self._count if self._count else 0

    @staticmethod
    def print_header():
        print('%-18s %12s %8s %12s %12s %12s' % ('Name', 'Total(s)', 'Count', 'Min(us)', 'Avg(us)', 'Max(us)'))

    def dump(self, name: str):
        print('%18s %12.3f %8d %12.3f %12.3f %12.3f' % (
            name, self._total_sec, self._count, self._min * 1e6, self.get_avg_sec() * 1e6, self._max * 1e6
        ))


ProfilerRegistry = defaultdict(Profiler)
