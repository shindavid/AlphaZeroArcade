from alphazero.logic.shutdown_manager import ShutdownManager

import threading
import unittest
from unittest.mock import patch


class TestShutdownManagerState(unittest.TestCase):

    def test_initial_state(self):
        sm = ShutdownManager()
        self.assertFalse(sm.shutdown_requested())
        self.assertTrue(sm.active())

    def test_request_shutdown(self):
        sm = ShutdownManager()
        sm.request_shutdown(0)
        self.assertTrue(sm.shutdown_requested())

    def test_active_before_shutdown(self):
        sm = ShutdownManager()
        sm.request_shutdown(0)
        # active() returns True until shutdown() is actually called
        self.assertTrue(sm.active())

    def test_request_shutdown_max_code(self):
        sm = ShutdownManager()
        sm.request_shutdown(1)
        sm.request_shutdown(3)
        sm.request_shutdown(2)
        self.assertEqual(sm._shutdown_code, 3)


class TestShutdownManagerActions(unittest.TestCase):

    @patch('sys.exit')
    def test_shutdown_calls_sys_exit(self, mock_exit):
        sm = ShutdownManager()
        sm.request_shutdown(42)
        sm.shutdown()
        mock_exit.assert_called_once_with(42)

    @patch('sys.exit')
    def test_shutdown_runs_actions_in_order(self, mock_exit):
        sm = ShutdownManager()
        results = []
        sm.register(lambda: results.append('first'), 'action1')
        sm.register(lambda: results.append('second'), 'action2')
        sm.request_shutdown(0)
        sm.shutdown()
        self.assertEqual(results, ['first', 'second'])

    @patch('sys.exit')
    def test_shutdown_continues_on_action_error(self, mock_exit):
        sm = ShutdownManager()
        results = []
        sm.register(lambda: (_ for _ in ()).throw(RuntimeError('fail')), 'broken_action')
        sm.register(lambda: results.append('ok'), 'good_action')
        sm.request_shutdown(0)
        sm.shutdown()
        self.assertEqual(results, ['ok'])
        mock_exit.assert_called_once()

    @patch('sys.exit')
    def test_shutdown_sets_in_progress(self, mock_exit):
        sm = ShutdownManager()
        sm.request_shutdown(0)
        sm.shutdown()
        self.assertFalse(sm.active())


class TestShutdownManagerRegister(unittest.TestCase):

    def test_register_action(self):
        sm = ShutdownManager()
        sm.register(lambda: None, 'test')
        self.assertEqual(len(sm._shutdown_actions), 1)

    def test_register_multiple(self):
        sm = ShutdownManager()
        sm.register(lambda: None, 'a')
        sm.register(lambda: None, 'b')
        sm.register(lambda: None, 'c')
        self.assertEqual(len(sm._shutdown_actions), 3)


class TestShutdownManagerThreadSafety(unittest.TestCase):

    def test_concurrent_request_shutdown(self):
        sm = ShutdownManager()
        barrier = threading.Barrier(10)

        def request(code):
            barrier.wait()
            sm.request_shutdown(code)

        threads = [threading.Thread(target=request, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertTrue(sm.shutdown_requested())
        self.assertEqual(sm._shutdown_code, 9)

    def test_wait_for_shutdown_request(self):
        sm = ShutdownManager()
        result = []

        def waiter():
            sm.wait_for_shutdown_request()
            result.append('done')

        t = threading.Thread(target=waiter)
        t.start()
        sm.request_shutdown(0)
        t.join(timeout=2)
        self.assertFalse(t.is_alive())
        self.assertEqual(result, ['done'])


if __name__ == '__main__':
    unittest.main()
