#!/usr/bin/env python3
from tools.one_off.upgrader_base import ColumnAdditionInstruction, ColumnRenameInstruction, \
    UpgraderBase

# Affected DB files that contain the mcts_agents table:
_MCTS_AGENTS_GLOBS = [
    '**/databases/evaluation/*.db',
    '**/databases/benchmark.db',
    '**/databases/ratings.db',
]


class Upgrader(UpgraderBase):
    FROM_VERSION = 8
    TO_VERSION = 9

    def get_instructions(self):
        instructions = []
        for glob in _MCTS_AGENTS_GLOBS:
            instructions.append(ColumnRenameInstruction(
                filename_glob=glob,
                table_name='mcts_agents',
                old_name='paradigm',
                new_name='spec_name',
            ))
            instructions.append(ColumnAdditionInstruction(
                filename_glob=glob,
                table_name='mcts_agents',
                column_name='extra_player_args',
                column_value='',
            ))
        return instructions


def main():
    upgrader = Upgrader()
    upgrader.run()


if __name__ == '__main__':
    main()
