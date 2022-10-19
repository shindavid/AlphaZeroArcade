from collections import defaultdict
import time


class Profiler:
    __slots__ = ['_total_sec', '_count', '_start_t']

    def __init__(self):
        self._total_sec = 0
        self._count = 0
        self._start_t = 0

    def start(self):
        self._start_t = time.time()

    def stop(self):
        self._total_sec += time.time() - self._start_t
        self._count += 1

    def get_total_sec(self):
        return self._total_sec

    def get_count(self):
        return self._count

    def get_avg_sec(self):
        return self._total_sec / self._count if self._count else 0

    @staticmethod
    def print_header():
        print('%-18s %12s %8s %12s' % ('Name', 'Total(s)', 'Count', 'Avg(us)'))

    def dump(self, name: str):
        print('%18s %12.3f %8d %12.3f' % (name, self.get_total_sec(), self.get_count(), self.get_avg_sec() * 1e6))


ProfilerRegistry = defaultdict(Profiler)
