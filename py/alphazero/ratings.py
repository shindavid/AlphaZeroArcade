import collections
from typing import Dict


def int_parse(s: str, prefix: str):
    assert s.startswith(prefix), s
    return int(s[len(prefix):])


class WinLossDrawCounts:
    def __init__(self, win=0, loss=0, draw=0):
        self.win = win
        self.loss = loss
        self.draw = draw

    def __iadd__(self, other):
        self.win += other.win
        self.loss += other.loss
        self.draw += other.draw
        return self

    def __str__(self):
        return f'W{self.win} L{self.loss} D{self.draw}'


class MatchRecord:
    def __init__(self):
        self.counts: Dict[int, WinLossDrawCounts] = collections.defaultdict(WinLossDrawCounts)

    def update(self, player_id: int, counts: WinLossDrawCounts):
        self.counts[player_id] += counts

    def get(self, player_id: int) -> WinLossDrawCounts:
        return self.counts[player_id]

    def empty(self) -> bool:
        return len(self.counts) == 0


def extract_match_record(stdout: str) -> MatchRecord:
    """
    ...
    All games complete!
    pid=0 name=foo W40 L24 D0 [40]
    pid=1 name=bar W24 L40 D0 [24]
    ...
    """
    record = MatchRecord()
    for line in stdout.splitlines():
        tokens = line.split()
        if len(tokens) > 1 and tokens[0].startswith('pid=') and tokens[0][4:].isdigit():
            player_id = int_parse(tokens[0], 'pid=')
            win = int_parse(tokens[2], 'W')
            loss = int_parse(tokens[3], 'L')
            draw = int_parse(tokens[4], 'D')
            counts = WinLossDrawCounts(win, loss, draw)
            record.update(player_id, counts)

    assert not record.empty(), stdout
    counts1 = record.get(0)
    counts2 = record.get(1)
    assert (counts1.win, counts1.loss, counts1.draw) == (counts2.loss, counts2.win, counts2.draw)
    return record
