from util.subprocess_util import defaultize_kwargs

import subprocess
import unittest


class TestDefaultizeKwargs(unittest.TestCase):

    def test_string_cmd_sets_shell_true(self):
        result = defaultize_kwargs('echo hello')
        self.assertTrue(result['shell'])

    def test_list_cmd_sets_shell_false(self):
        result = defaultize_kwargs(['echo', 'hello'])
        self.assertFalse(result['shell'])

    def test_default_stdout(self):
        result = defaultize_kwargs('echo hello')
        self.assertEqual(result['stdout'], subprocess.PIPE)

    def test_default_stderr(self):
        result = defaultize_kwargs('echo hello')
        self.assertEqual(result['stderr'], subprocess.PIPE)

    def test_default_stdin(self):
        result = defaultize_kwargs('echo hello')
        self.assertEqual(result['stdin'], subprocess.PIPE)

    def test_default_encoding(self):
        result = defaultize_kwargs('echo hello')
        self.assertEqual(result['encoding'], 'utf-8')

    def test_user_override_shell(self):
        result = defaultize_kwargs('echo hello', shell=False)
        self.assertFalse(result['shell'])

    def test_user_override_stdout(self):
        result = defaultize_kwargs('echo hello', stdout=None)
        self.assertIsNone(result['stdout'])

    def test_user_override_encoding(self):
        result = defaultize_kwargs('echo hello', encoding='ascii')
        self.assertEqual(result['encoding'], 'ascii')

    def test_extra_kwargs_preserved(self):
        result = defaultize_kwargs('echo hello', cwd='/tmp')
        self.assertEqual(result['cwd'], '/tmp')

    def test_all_defaults_present(self):
        result = defaultize_kwargs(['ls'])
        expected_keys = {'shell', 'stdout', 'stdin', 'stderr', 'encoding'}
        self.assertTrue(expected_keys.issubset(result.keys()))


if __name__ == '__main__':
    unittest.main()
