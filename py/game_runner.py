import random
from typing import List, Type

from interface import AbstractPlayer, AbstractGameState, GameResult


class GameRunner:
    def __init__(self, state_cls: Type[AbstractGameState], players: List[AbstractPlayer]):
        self.state_cls = state_cls
        self.players = list(players)

    def randomize_player_order(self):
        random.shuffle(self.players)

    def run(self) -> GameResult:
        for p, player in enumerate(self.players):
            player.start_game(self.players, p)

        state = self.state_cls()

        result = None
        while result is None:
            player_index = state.get_current_player()
            player = self.players[player_index]

            valid_actions = state.get_valid_actions()
            action = player.get_action(state, valid_actions)
            assert valid_actions[action]
            result = state.apply_move(action)

            for p in self.players:
                p.receive_state_change(player_index, state, action, result)

        return result
