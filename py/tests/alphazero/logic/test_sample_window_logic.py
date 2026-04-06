import unittest
from alphazero.logic.sample_window_logic import Window, get_required_dataset_size
from shared.training_params import TrainingParams


def make_params(window_size=100, target_sample_rate=8, minibatch_size=10, minibatches_per_epoch=10):
    """Build a TrainingParams with a fixed window size for predictable tests."""
    return TrainingParams(
        window_size_function_str=f'fixed({window_size})',
        target_sample_rate=target_sample_rate,
        minibatch_size=minibatch_size,
        minibatches_per_epoch=minibatches_per_epoch,
    )


class TestGetRequiredDatasetSize(unittest.TestCase):
    def test_returns_b_when_condition_already_satisfied(self):
        # With r < t, the condition lhs <= rhs(b) is easy to satisfy.
        # Window with low sample_rate: existing samples are cheap, rhs(b) is large.
        params = make_params(window_size=100, target_sample_rate=8)
        # samples_per_window = 10 * 10 = 100
        # lhs = b*r + s = 50*2 + 100 = 200
        # rhs(b=50) = 50*2 + (8-2)*100 = 100 + 600 = 700  => lhs <= rhs => return b
        window = Window(start=0, end=50, sample_rate=2.0)
        result = get_required_dataset_size(params, window)
        self.assertEqual(result, 50)

    def test_returns_larger_n_when_sample_rate_exceeds_target(self):
        # With r > t, rhs(n) shrinks as window_size term is negative.
        # r=10 > t=8 means each extra position adds less value than it costs.
        params = make_params(window_size=100, target_sample_rate=8)
        # lhs = 50*10 + 100 = 600
        # rhs(n) = n*10 + (8-10)*100 = 10n - 200
        # Need 10n - 200 >= 600, i.e. n >= 80
        window = Window(start=0, end=50, sample_rate=10.0)
        result = get_required_dataset_size(params, window)
        self.assertEqual(result, 80)

    def test_result_is_at_least_b(self):
        params = make_params()
        window = Window(start=0, end=200, sample_rate=1.0)
        result = get_required_dataset_size(params, window)
        self.assertGreaterEqual(result, window.end)

    def test_higher_sample_rate_requires_more_data(self):
        # Everything else equal, a higher previous sample_rate requires more future data
        params = make_params(window_size=100, target_sample_rate=8)
        window_low = Window(start=0, end=50, sample_rate=5.0)
        window_high = Window(start=0, end=50, sample_rate=12.0)
        result_low = get_required_dataset_size(params, window_low)
        result_high = get_required_dataset_size(params, window_high)
        self.assertLessEqual(result_low, result_high)

    def test_larger_previous_window_end_shifts_result(self):
        # A larger b means more positions need to be covered, so result >= b increases
        params = make_params(window_size=100, target_sample_rate=8)
        window_small = Window(start=0, end=50, sample_rate=10.0)
        window_large = Window(start=0, end=200, sample_rate=10.0)
        result_small = get_required_dataset_size(params, window_small)
        result_large = get_required_dataset_size(params, window_large)
        self.assertLessEqual(result_small, result_large)


if __name__ == '__main__':
    unittest.main()
