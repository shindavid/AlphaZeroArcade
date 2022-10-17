import os
import random
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config
from interface import AbstractPlayer, PlayerIndex, ActionIndex, GameResult, ActionMask
from connect4.game_logic import C4GameState, NUM_COLUMNS


@dataclass
class PerfectPlayerParams:
    c4_solver_dir: Optional[str] = Config.instance().get('c4.solver_dir')
    strong: bool = True


class PerfectPlayer(AbstractPlayer):
    """
    In winning positions, randomly chooses a winning move. If in strong mode, then only considers moves that win
    quickest.

    In losing positions, randomly chooses a move that loses slowest.

    In drawn positions, randomly chooses a drawing move.
    """
    def __init__(self, params: PerfectPlayerParams):
        c4_solver_dir = os.path.expanduser(params.c4_solver_dir)
        c4_solver_bin = os.path.join(c4_solver_dir, 'c4solver')
        c4_solver_book = os.path.join(c4_solver_dir, '7x6.book')
        assert os.path.isdir(c4_solver_dir)
        assert os.path.isfile(c4_solver_bin)
        assert os.path.isfile(c4_solver_book)
        c4_cmd = f"{c4_solver_bin} -b {c4_solver_book} -a"
        proc = subprocess.Popen(c4_cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE,
                                stderr=subprocess.PIPE, encoding='utf-8')
        self.proc = proc
        self.strong_mode = params.strong
        self.move_history = ''

    def get_name(self) -> str:
        return 'Perfect'

    def start_game(self, players: List[AbstractPlayer], seat_assignment: PlayerIndex):
        pass

    def receive_state_change(self, p: PlayerIndex, state: C4GameState,
                             action_index: ActionIndex, result: GameResult):
        column = action_index + 1
        self.move_history += str(column)

    def get_action(self, state: C4GameState, valid_actions: ActionMask) -> ActionIndex:
        self.proc.stdin.write(self.move_history + '\n')
        self.proc.stdin.flush()
        stdout = self.proc.stdout.readline()

        move_scores = list(map(int, stdout.split()[-NUM_COLUMNS:]))
        best_score = max(move_scores)

        if self.strong_mode or best_score <= 0:
            move_arr = (np.array(move_scores) == best_score)
        else:
            move_arr = np.array(move_scores) > 0

        return random.choice(np.where(move_arr)[0])
