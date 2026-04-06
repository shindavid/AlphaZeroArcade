from alphazero.logic.run_params import RunParams

import unittest


class TestIsValidTag(unittest.TestCase):

    def test_simple_tag(self):
        self.assertTrue(RunParams.is_valid_tag('v1'))

    def test_alphanumeric(self):
        self.assertTrue(RunParams.is_valid_tag('my_run_123'))

    def test_hyphenated(self):
        self.assertTrue(RunParams.is_valid_tag('my-run'))

    def test_empty_string(self):
        self.assertFalse(RunParams.is_valid_tag(''))

    def test_at_sign(self):
        self.assertFalse(RunParams.is_valid_tag('v1@host'))

    def test_period(self):
        self.assertFalse(RunParams.is_valid_tag('v1.0'))

    def test_path_traversal(self):
        self.assertFalse(RunParams.is_valid_tag('../etc'))

    def test_slash(self):
        self.assertFalse(RunParams.is_valid_tag('a/b'))

    def test_single_char(self):
        self.assertTrue(RunParams.is_valid_tag('x'))


class TestAddToCmd(unittest.TestCase):

    def test_basic(self):
        rp = RunParams(game='c4', tag='v1')
        cmd = []
        rp.add_to_cmd(cmd)
        self.assertEqual(cmd, ['--game', 'c4', '--tag', 'v1'])

    def test_different_game(self):
        rp = RunParams(game='tictactoe', tag='test')
        cmd = ['python', 'script.py']
        rp.add_to_cmd(cmd)
        self.assertIn('--game', cmd)
        self.assertIn('tictactoe', cmd)
        self.assertIn('--tag', cmd)
        self.assertIn('test', cmd)


if __name__ == '__main__':
    unittest.main()
