from shared.basic_types import SearchParadigm, ShapeInfo, ShapeInfoCollection

import unittest


class TestSearchParadigm(unittest.TestCase):

    def test_alpha0_valid(self):
        self.assertTrue(SearchParadigm.is_valid('alpha0'))

    def test_beta0_valid(self):
        self.assertTrue(SearchParadigm.is_valid('beta0'))

    def test_invalid_string(self):
        self.assertFalse(SearchParadigm.is_valid('gamma0'))

    def test_empty_string(self):
        self.assertFalse(SearchParadigm.is_valid(''))

    def test_case_sensitive(self):
        self.assertFalse(SearchParadigm.is_valid('Alpha0'))

    def test_enum_values(self):
        self.assertEqual(SearchParadigm.AlphaZero.value, 'alpha0')
        self.assertEqual(SearchParadigm.BetaZero.value, 'beta0')


class TestShapeInfo(unittest.TestCase):

    def test_creation(self):
        si = ShapeInfo(name='policy', target_index=0, shape=(3, 3))
        self.assertEqual(si.name, 'policy')
        self.assertEqual(si.target_index, 0)
        self.assertEqual(si.shape, (3, 3))

    def test_different_shapes(self):
        si = ShapeInfo(name='value', target_index=1, shape=(1,))
        self.assertEqual(si.shape, (1,))


class TestShapeInfoCollection(unittest.TestCase):

    def test_creation(self):
        input_shapes = {'board': ShapeInfo('board', 0, (3, 3, 2))}
        target_shapes = {'policy': ShapeInfo('policy', 0, (9,))}
        head_shapes = {'value': ShapeInfo('value', 1, (1,))}

        collection = ShapeInfoCollection(
            input_shapes=input_shapes,
            target_shapes=target_shapes,
            head_shapes=head_shapes,
        )
        self.assertIn('board', collection.input_shapes)
        self.assertIn('policy', collection.target_shapes)
        self.assertIn('value', collection.head_shapes)

    def test_empty_dicts(self):
        collection = ShapeInfoCollection(
            input_shapes={},
            target_shapes={},
            head_shapes={},
        )
        self.assertEqual(len(collection.input_shapes), 0)


if __name__ == '__main__':
    unittest.main()
