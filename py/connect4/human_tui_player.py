import os
import sys
from typing import List, Optional

from connect4.game_logic import C4GameState

sys.path.append(os.path.join(sys.path[0], '..'))
from interface import AbstractPlayer, PlayerIndex, ActionIndex, GameResult, ActionMask


class C4HumanTuiPlayer(AbstractPlayer):
    def __init__(self):
        self.player_names = ['?', '?']
        self.my_index = None
        self.last_action: Optional[ActionIndex] = None

    def get_name(self) -> str:
        return 'Human'

    def start_game(self, players: List[AbstractPlayer], seat_assignment: PlayerIndex):
        self.player_names = [p.get_name() for p in players]
        self.my_index = seat_assignment

    def receive_state_change(self, p: PlayerIndex, state: C4GameState,
                             action_index: ActionIndex, result: GameResult):
        self.last_action = action_index
        if self.my_index != p:
            self.print_state(state)

    def print_state(self, state: C4GameState):
        column = self.last_action + 1
        os.system('clear')
        print(state.to_ascii_drawing(add_legend=True, player_names=self.player_names, highlight_column=column))

    def get_action(self, state: C4GameState, valid_actions: ActionMask) -> ActionIndex:
        my_action = None
        while True:
            if my_action is not None:
                self.print_state(state)
                print(f'Invalid input!')
            my_action = input('Enter move [1-7]: ')
            try:
                my_action = int(my_action) - 1
                assert valid_actions[my_action]
                break
            except:
                continue

        return my_action
