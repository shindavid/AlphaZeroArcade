#!/usr/bin/env python3
"""
Pit two players against each other.
"""
import time
from collections import defaultdict
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from game_runner import GameRunner
from connect4.game_logic import C4GameState
from connect4.nnet_player import NNetPlayer, NNetPlayerParams
from connect4.perfect_player import PerfectPlayer, PerfectPlayerParams


def main():
    use_perfect = False
    num_games = 5
    num_mcts_iters = 100

    if use_perfect:
        cpu1 = PerfectPlayer(PerfectPlayerParams())
        cpu1.set_name('Perfect')
    else:
        params1 = NNetPlayerParams(num_mcts_iters=num_mcts_iters)
        cpu1 = NNetPlayer(params1)
        cpu1.set_name('MCTS1-' + str(params1.num_mcts_iters))

    params2 = NNetPlayerParams(num_mcts_iters=num_mcts_iters)
    cpu2 = NNetPlayer(params2)
    cpu2.set_name('MCTS2-' + str(params2.num_mcts_iters))

    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # name -> color -> W/L/D -> count

    def stats_str(name, color):
        counts = stats[name][color]
        w = counts[1]
        l = counts[0]
        d = counts[0.5]
        return f'{w}W {l}L {d}D'

    results_str_dict = {
        0: 'Y wins',
        0.5: 'draw  ',
        1: 'R wins',
    }

    r = range(1, num_games, 2) if use_perfect else range(num_games)
    runtimes = []
    for n in r:
        players = [None, None]
        m = n % 2
        players[m] = cpu1
        players[1-m] = cpu2

        t1 = time.time()
        runner = GameRunner(C4GameState, players)
        result = runner.run()
        t2 = time.time()
        runtimes.append(t2 - t1)

        for c in (0, 1):
            stats[players[c].get_name()][c][result[c]] += 1

        result_str = results_str_dict[result[0]]
        cumulative_result_str = ' '.join([
            f'R:{players[0].get_name()}:[{stats_str(players[0].get_name(), 0)}]',
            f'Y:{players[1].get_name()}:[{stats_str(players[1].get_name(), 1)}]',
        ])
        print(f'R:{players[0].get_name()} Y:{players[1].get_name()} Res:{result_str} {cumulative_result_str}')

    print('Final results:')
    for name, stats_by_color in stats.items():
        rw = stats_by_color[0][1]
        rl = stats_by_color[0][0]
        rd = stats_by_color[0][0.5]
        yw = stats_by_color[1][1]
        yl = stats_by_color[1][0]
        yd = stats_by_color[1][0.5]
        ow = rw + yw
        ol = rl + yl
        od = rd + yd
        print(f'{name}:')
        print(f'  as red:     {rw}W {rl}L {rd}D')
        print(f'  as yellow:  {yw}W {yl}L {yd}D')
        print(f'  as overall: {ow}W {ol}L {od}D')

    avg_runtime = sum(runtimes) / len(runtimes)
    max_runtime = max(runtimes)
    min_runtime = min(runtimes)
    print('Avg runtime: %.3fs' % avg_runtime)
    print('Max runtime: %.3fs' % max_runtime)
    print('Min runtime: %.3fs' % min_runtime)


if __name__ == '__main__':
    main()
