from shared.training_params import (
    FixedWindowSizeFunction,
    KataGoWindowSizeFunction,
    TrainingParams,
    WindowSizeFunction,
)

import unittest


class TestFixedWindowSizeFunction(unittest.TestCase):

    def test_returns_constant(self):
        f = FixedWindowSizeFunction(window_size=100)
        self.assertEqual(f(0), 100)
        self.assertEqual(f(50), 100)
        self.assertEqual(f(1000), 100)

    def test_different_sizes(self):
        f = FixedWindowSizeFunction(window_size=500)
        self.assertEqual(f(999), 500)


class TestKataGoWindowSizeFunction(unittest.TestCase):

    def test_at_n0(self):
        f = KataGoWindowSizeFunction(n0=1000)
        # f(n0) should equal n0 (the function is rescaled so f(n0)=n0)
        self.assertAlmostEqual(f(1000), 1000, places=0)

    def test_at_zero(self):
        f = KataGoWindowSizeFunction(n0=1000)
        result = f(0)
        # f(0) should be small / <= 0
        self.assertLessEqual(result, 1000)

    def test_never_exceeds_n(self):
        f = KataGoWindowSizeFunction(n0=1000)
        for n in [100, 500, 1000, 5000, 10000]:
            self.assertLessEqual(f(n), n)

    def test_monotonically_increasing(self):
        f = KataGoWindowSizeFunction(n0=1000)
        prev = f(1)
        for n in range(2, 5000, 100):
            curr = f(n)
            self.assertGreaterEqual(curr, prev)
            prev = curr

    def test_sublinear_growth(self):
        # For large n >> n0, the window should grow sub-linearly
        f = KataGoWindowSizeFunction(n0=1000)
        # f(10000) should be much less than 10000
        self.assertLess(f(10000), 10000)
        # But still significantly more than f(1000) = 1000
        self.assertGreater(f(10000), 1000)

    def test_custom_params(self):
        f = KataGoWindowSizeFunction(n0=500, alpha=0.75, beta=0.4)
        self.assertAlmostEqual(f(500), 500, places=0)


class TestWindowSizeFunctionCreate(unittest.TestCase):

    def test_create_fixed(self):
        f = WindowSizeFunction.create('fixed(window_size=200)', {})
        self.assertIsInstance(f, FixedWindowSizeFunction)
        self.assertEqual(f(999), 200)

    def test_create_katago(self):
        f = WindowSizeFunction.create('katago(n0=1000)', {})
        self.assertIsInstance(f, KataGoWindowSizeFunction)

    def test_create_with_eval_dict(self):
        f = WindowSizeFunction.create('fixed(window_size=x*2)', {'x': 50})
        self.assertEqual(f(0), 100)

    def test_create_invalid_raises(self):
        with self.assertRaises(ValueError):
            WindowSizeFunction.create('nonexistent()', {})

    def test_repr_set(self):
        s = 'fixed(window_size=200)'
        f = WindowSizeFunction.create(s, {})
        self.assertEqual(repr(f), s)


class TestTrainingParams(unittest.TestCase):

    def test_default_construction(self):
        tp = TrainingParams()
        self.assertIsNotNone(tp.window_size_function)
        self.assertEqual(tp.target_sample_rate, 8)
        self.assertEqual(tp.minibatches_per_epoch, 2048)
        self.assertEqual(tp.minibatch_size, 256)

    def test_samples_per_window(self):
        tp = TrainingParams(minibatch_size=128, minibatches_per_epoch=100)
        self.assertEqual(tp.samples_per_window(), 128 * 100)

    def test_window_function_created_from_str(self):
        tp = TrainingParams(window_size_function_str='fixed(window_size=500)')
        self.assertIsInstance(tp.window_size_function, FixedWindowSizeFunction)
        self.assertEqual(tp.window_size_function(999), 500)

    def test_add_to_cmd_no_diff(self):
        tp = TrainingParams()
        cmd = []
        tp.add_to_cmd(cmd)
        self.assertEqual(cmd, [])

    def test_add_to_cmd_with_diff(self):
        tp = TrainingParams(minibatch_size=512)
        cmd = []
        tp.add_to_cmd(cmd)
        self.assertIn('--minibatch-size', cmd)
        self.assertIn('512', cmd)


if __name__ == '__main__':
    unittest.main()
