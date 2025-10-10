#!/usr/bin/env python3
from tools.one_off.upgrader_base import ColumnAdditionInstruction, UpgraderBase


class Upgrader(UpgraderBase):
    FROM_VERSION = 7
    TO_VERSION = 8

    def get_instructions(self):
        return [ColumnAdditionInstruction(
            filename_glob='**/databases/evaluation/*.db',
            table_name=t,
            column_name='rating_tag',
            column_value=''
        ) for t in ('matches', 'evaluator_ratings')]


def main():
    upgrader = Upgrader()
    upgrader.run()


if __name__ == '__main__':
    main()
