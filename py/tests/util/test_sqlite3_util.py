from util.sqlite3_util import escape_value, DatabaseConnectionPool

import os
import sqlite3
import tempfile
import unittest


class TestEscapeValue(unittest.TestCase):

    def test_none(self):
        self.assertEqual(escape_value(None), 'NULL')

    def test_int(self):
        self.assertEqual(escape_value(42), '42')

    def test_negative_int(self):
        self.assertEqual(escape_value(-7), '-7')

    def test_float(self):
        self.assertEqual(escape_value(3.14), '3.14')

    def test_zero(self):
        self.assertEqual(escape_value(0), '0')

    def test_string(self):
        self.assertEqual(escape_value('hello'), "'hello'")

    def test_string_with_single_quote(self):
        self.assertEqual(escape_value("it's"), "'it''s'")

    def test_string_with_multiple_quotes(self):
        self.assertEqual(escape_value("a'b'c"), "'a''b''c'")

    def test_empty_string(self):
        self.assertEqual(escape_value(''), "''")

    def test_string_with_special_chars(self):
        self.assertEqual(escape_value('hello world'), "'hello world'")


class TestDatabaseConnectionPool(unittest.TestCase):

    def setUp(self):
        self.tmp_fd, self.tmp_path = tempfile.mkstemp(suffix='.db')
        os.close(self.tmp_fd)
        # Remove the file so the pool can create it fresh
        os.unlink(self.tmp_path)

    def tearDown(self):
        if os.path.exists(self.tmp_path):
            os.unlink(self.tmp_path)

    def test_create_db_with_create_cmds(self):
        create_cmds = ['CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)']
        pool = DatabaseConnectionPool(self.tmp_path, create_cmds)
        conn = pool.get_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO test (id, name) VALUES (1, 'alice')")
        conn.commit()
        cursor.execute("SELECT name FROM test WHERE id = 1")
        row = cursor.fetchone()
        self.assertEqual(row[0], 'alice')
        pool.close_connections()

    def test_get_connection_returns_same_for_same_thread(self):
        create_cmds = ['CREATE TABLE t (id INTEGER PRIMARY KEY)']
        pool = DatabaseConnectionPool(self.tmp_path, create_cmds)
        c1 = pool.get_connection()
        c2 = pool.get_connection()
        self.assertIs(c1, c2)
        pool.close_connections()

    def test_close_connections(self):
        create_cmds = ['CREATE TABLE t (id INTEGER PRIMARY KEY)']
        pool = DatabaseConnectionPool(self.tmp_path, create_cmds)
        pool.get_connection()
        pool.close_connections()
        # After close, getting connection again should create a new one
        c = pool.get_connection()
        self.assertIsNotNone(c)
        pool.close_connections()

    def test_no_create_cmds_no_file_raises(self):
        pool = DatabaseConnectionPool(self.tmp_path, create_cmds=None)
        with self.assertRaises(ValueError):
            pool.get_connection()

    def test_existing_db_no_create_cmds(self):
        # Create DB first
        conn = sqlite3.connect(self.tmp_path)
        conn.execute('CREATE TABLE t (id INTEGER PRIMARY KEY)')
        conn.commit()
        conn.close()

        # Now open without create_cmds - should work fine
        pool = DatabaseConnectionPool(self.tmp_path)
        c = pool.get_connection()
        self.assertIsNotNone(c)
        pool.close_connections()


if __name__ == '__main__':
    unittest.main()
