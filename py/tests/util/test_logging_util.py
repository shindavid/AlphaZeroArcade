from util.logging_util import LoggingParams, CustomFormatter, configure_logger

import argparse
import datetime
import logging
import os
import tempfile
import time
import unittest

import util.logging_util as logging_util


class TestLoggingParams(unittest.TestCase):

    def test_create_from_args(self):
        args = argparse.Namespace(debug=True, debug_module=['mod1', 'mod2'])
        params = LoggingParams.create(args)
        self.assertTrue(params.debug)
        self.assertEqual(params.debug_module, ['mod1', 'mod2'])

    def test_create_debug_false(self):
        args = argparse.Namespace(debug=False, debug_module=[])
        params = LoggingParams.create(args)
        self.assertFalse(params.debug)

    def test_create_coerces_to_bool(self):
        args = argparse.Namespace(debug=0, debug_module=[])
        params = LoggingParams.create(args)
        self.assertFalse(params.debug)

    def test_add_to_cmd_empty(self):
        params = LoggingParams(debug=False, debug_module=[])
        cmd = []
        params.add_to_cmd(cmd)
        self.assertEqual(cmd, [])

    def test_add_to_cmd_debug(self):
        params = LoggingParams(debug=True)
        cmd = []
        params.add_to_cmd(cmd)
        self.assertEqual(cmd, ['--debug'])

    def test_add_to_cmd_debug_modules(self):
        params = LoggingParams(debug=False, debug_module=['mod1', 'mod2'])
        cmd = []
        params.add_to_cmd(cmd)
        self.assertEqual(cmd, ['--debug-module', 'mod1', 'mod2'])

    def test_add_to_cmd_debug_and_modules(self):
        params = LoggingParams(debug=True, debug_module=['mymod'])
        cmd = []
        params.add_to_cmd(cmd)
        self.assertEqual(cmd, ['--debug', '--debug-module', 'mymod'])

    def test_add_args(self):
        parser = argparse.ArgumentParser()
        LoggingParams.add_args(parser)
        args = parser.parse_args(['--debug', '--debug-module', 'a', 'b'])
        self.assertTrue(args.debug)
        self.assertEqual(args.debug_module, ['a', 'b'])

    def test_add_args_defaults(self):
        parser = argparse.ArgumentParser()
        LoggingParams.add_args(parser)
        args = parser.parse_args([])
        self.assertFalse(args.debug)
        self.assertEqual(args.debug_module, [])

    def test_default_debug_module_empty_list(self):
        params = LoggingParams(debug=True)
        self.assertEqual(params.debug_module, [])


class TestCustomFormatter(unittest.TestCase):

    def _make_record(self):
        record = logging.LogRecord(
            name='test', level=logging.INFO, pathname='', lineno=0,
            msg='hello', args=(), exc_info=None,
        )
        return record

    def test_format_time_contains_microseconds(self):
        formatter = CustomFormatter('%(asctime)s %(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S.%f')
        record = self._make_record()
        formatted = formatter.formatTime(record, '%Y-%m-%d %H:%M:%S.%f')
        # Should contain a '.' followed by 6 digits (microseconds)
        parts = formatted.split('.')
        self.assertEqual(len(parts), 2)
        self.assertEqual(len(parts[1]), 6)

    def test_format_time_matches_datetime(self):
        formatter = CustomFormatter('%(asctime)s %(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S.%f')
        record = self._make_record()
        formatted = formatter.formatTime(record, '%Y-%m-%d %H:%M:%S.%f')

        # Verify it parses back correctly
        dt = datetime.datetime.strptime(formatted, '%Y-%m-%d %H:%M:%S.%f')
        self.assertIsInstance(dt, datetime.datetime)

    def test_format_time_uses_record_created(self):
        record = self._make_record()
        # Override created to a known timestamp
        record.created = 1700000000.123456  # 2023-11-14 22:13:20.123456 UTC

        formatter = CustomFormatter()
        formatted = formatter.formatTime(record, '%H:%M:%S.%f')
        self.assertIn('123456', formatted)


class TestConfigureLogger(unittest.TestCase):

    def setUp(self):
        self._saved_handlers = list(logging.root.handlers)
        self._saved_level = logging.root.level
        self._saved_listener = logging_util._listener

    def tearDown(self):
        # Stop any listener we started
        if logging_util._listener and logging_util._listener is not self._saved_listener:
            try:
                logging_util._listener.stop()
            except Exception:
                pass

        # Restore original state
        logging_util._listener = self._saved_listener

        for h in list(logging.root.handlers):
            logging.root.removeHandler(h)
        for h in self._saved_handlers:
            logging.root.addHandler(h)
        logging.root.setLevel(self._saved_level)

    def test_default_level_is_info(self):
        configure_logger()
        self.assertEqual(logging.root.level, logging.INFO)

    def test_debug_level_when_requested(self):
        params = LoggingParams(debug=True)
        configure_logger(params=params)
        self.assertEqual(logging.root.level, logging.DEBUG)

    def test_logs_to_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            tmp_path = f.name

        try:
            configure_logger(filename=tmp_path, mode='w')
            test_logger = logging.getLogger('test_configure')
            test_logger.info('test message 12345')

            # Give QueueListener time to process
            time.sleep(0.2)

            with open(tmp_path, 'r') as f:
                content = f.read()
            self.assertIn('test message 12345', content)
        finally:
            os.unlink(tmp_path)

    def test_per_module_debug(self):
        params = LoggingParams(debug=False, debug_module=['my_special_module'])
        configure_logger(params=params)

        # Root should be INFO
        self.assertEqual(logging.root.level, logging.INFO)
        # But the specific module logger should be DEBUG
        mod_logger = logging.getLogger('my_special_module')
        self.assertEqual(mod_logger.level, logging.DEBUG)


if __name__ == '__main__':
    unittest.main()
