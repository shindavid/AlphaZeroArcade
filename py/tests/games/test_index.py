from games.index import get_game_spec, is_valid_game_name, ALL_GAME_SPECS, GAME_SPECS_BY_NAME

import unittest


class TestIsValidGameName(unittest.TestCase):

    def test_all_registered_games(self):
        for spec in ALL_GAME_SPECS:
            self.assertTrue(is_valid_game_name(spec.name), f'{spec.name} should be valid')

    def test_invalid_game(self):
        self.assertFalse(is_valid_game_name('nonexistent_game'))

    def test_empty_string(self):
        self.assertFalse(is_valid_game_name(''))

    def test_known_games(self):
        known = ['c4', 'tictactoe', 'hex', 'othello', 'chess']
        for name in known:
            self.assertTrue(is_valid_game_name(name), f'{name} should be valid')


class TestGetGameSpec(unittest.TestCase):

    def test_valid_game(self):
        spec = get_game_spec('c4')
        self.assertEqual(spec.name, 'c4')

    def test_tictactoe(self):
        spec = get_game_spec('tictactoe')
        self.assertEqual(spec.name, 'tictactoe')

    def test_invalid_raises_value_error(self):
        with self.assertRaises(ValueError):
            get_game_spec('nonexistent')

    def test_each_spec_returns_correct_type(self):
        for name, spec in GAME_SPECS_BY_NAME.items():
            retrieved = get_game_spec(name)
            self.assertIs(retrieved, spec)


class TestAllGameSpecs(unittest.TestCase):

    def test_no_duplicates(self):
        names = [spec.name for spec in ALL_GAME_SPECS]
        self.assertEqual(len(names), len(set(names)))

    def test_specs_by_name_consistent(self):
        self.assertEqual(len(ALL_GAME_SPECS), len(GAME_SPECS_BY_NAME))

    def test_each_spec_has_name(self):
        for spec in ALL_GAME_SPECS:
            self.assertTrue(spec.name, f'Spec {spec} has empty name')

    def test_each_spec_has_model_configs(self):
        for spec in ALL_GAME_SPECS:
            configs = spec.model_configs
            self.assertIsInstance(configs, dict)
            self.assertGreater(len(configs), 0, f'{spec.name} has no model configs')


if __name__ == '__main__':
    unittest.main()
