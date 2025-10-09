#!/usr/bin/env python3
from tools.one_off.upgrader_base import UpgraderBase, DatabaseTable


class Upgrader(UpgraderBase):
    FROM_VERSION = 6
    TO_VERSION = 7

    def rewrite_db_table(self, table: DatabaseTable):
        if table.db_file.name.endswith('databases/benchmark.db'):
            if table.name == 'mcts_agents':
                table.schedule_column_add('paradigm', 'alpha0')


def main():
    upgrader = Upgrader()
    upgrader.run()


if __name__ == '__main__':
    main()
