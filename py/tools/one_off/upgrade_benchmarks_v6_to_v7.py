#!/usr/bin/env python3
from tools.one_off.upgrader_base import ColumnAdditionInstruction, UpgraderBase


class Upgrader(UpgraderBase):
    FROM_VERSION = 6
    TO_VERSION = 7

    def get_instructions(self):
        return [ColumnAdditionInstruction(
            filename_glob='**/databases/benchmark.db',
            table_name='mcts_agents',
            column_name='paradigm',
            column_value='alpha0'
        )]


def main():
    upgrader = Upgrader()
    upgrader.run()


if __name__ == '__main__':
    main()
