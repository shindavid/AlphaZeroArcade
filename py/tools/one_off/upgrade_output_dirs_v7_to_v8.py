#!/usr/bin/env python3
from tools.one_off.upgrader_base import UpgraderBase, DatabaseTable


class Upgrader(UpgraderBase):
    FROM_VERSION = 7
    TO_VERSION = 8

    def rewrite_db_table(self, table: DatabaseTable):
        if table.is_evaluation_db():
            if table.name in ('matches', 'evaluator_ratings'):
                table.schedule_column_add('rating_tag', '')


def main():
    upgrader = Upgrader()
    upgrader.run()


if __name__ == '__main__':
    main()
