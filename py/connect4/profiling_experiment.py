#!/usr/bin/env python3
"""
Pit two players against each other.
"""
import argparse
import time
from collections import defaultdict

from game_runner import GameRunner
from profiling import ProfilerRegistry, Profiler

from connect4.game_logic import C4GameState
from connect4.nnet_player import NNetPlayer, NNetPlayerParams
from connect4.perfect_player import PerfectPlayer, PerfectPlayerParams
from util.repo_util import Repo


class Args:
    use_perfect = False
    num_games = 2
    num_mcts_iters = 400
    model_file = None
    debug_file = None

    @staticmethod
    def load(args):
        Args.use_perfect = args.use_perfect
        Args.num_games = args.num_games
        Args.num_mcts_iters = args.num_mcts_iters
        Args.model_file = args.model_file
        Args.debug_file = args.debug_file


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--use-perfect", action='store_true', help='mcts vs perfect')
    parser.add_argument("-g", "--num-games", default=Args.num_games, type=int,
                        help='num games (default: %(default)s)')
    parser.add_argument("-i", "--num-mcts-iters", default=Args.num_mcts_iters, type=int,
                        help='num num iterations (default: %(default)s)')
    parser.add_argument('-m', '--model-file', default=Repo.c4_model(), help='c4 model (default: %(default)s)')
    parser.add_argument('-d', '--debug-file', help='debug file (for first player)')

    args = parser.parse_args()
    Args.load(args)


def main():
    load_args()

    use_perfect = Args.use_perfect
    num_games = Args.num_games
    num_mcts_iters = Args.num_mcts_iters
    model_file = Args.model_file
    debug_file = Args.debug_file

    params1 = NNetPlayerParams(num_mcts_iters=num_mcts_iters, model_file=model_file, debug_filename=debug_file)
    cpu1 = NNetPlayer(params1)
    cpu1.set_name('MCTS1-' + str(params1.num_mcts_iters))

    if use_perfect:
        cpu2 = PerfectPlayer(PerfectPlayerParams())
        cpu2.set_name('Perfect')
    else:
        params2 = NNetPlayerParams(num_mcts_iters=num_mcts_iters, model_file=model_file)
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

    r = range(0, 2 * num_games, 2) if use_perfect else range(num_games)
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

    Profiler.print_header()
    for key, profiler in ProfilerRegistry.items():
        profiler.dump(key)


if __name__ == '__main__':
    main()
