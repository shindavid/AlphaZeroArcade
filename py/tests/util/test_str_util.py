import unittest
from util.str_util import rreplace, center_text, inject_arg, inject_args, make_args_str


class TestRreplace(unittest.TestCase):
    def test_replaces_from_back(self):
        self.assertEqual(rreplace('aXbXcX', 'X', 'Y', 1), 'aXbXcY')

    def test_count_two(self):
        self.assertEqual(rreplace('aXbXcX', 'X', 'Y', 2), 'aXbYcY')

    def test_count_exceeds_occurrences(self):
        # count larger than actual occurrences replaces all
        self.assertEqual(rreplace('aXbX', 'X', 'Y', 10), 'aYbY')

    def test_no_match(self):
        self.assertEqual(rreplace('abc', 'Z', 'Y', 1), 'abc')

    def test_empty_string(self):
        self.assertEqual(rreplace('', 'X', 'Y', 1), '')


class TestCenterText(unittest.TestCase):
    def test_even_padding_even_total(self):
        # padding m=2, a=1, b=1
        self.assertEqual(center_text('abc', 5), ' abc ')

    def test_odd_extra_padding_goes_right(self):
        # m=3, a=1, b=2
        self.assertEqual(center_text('abc', 6), ' abc  ')

    def test_larger_odd_total(self):
        # m=4, a=2, b=2
        self.assertEqual(center_text('abc', 7), '  abc  ')

    def test_exact_length(self):
        self.assertEqual(center_text('abc', 3), 'abc')

    def test_single_extra_goes_right(self):
        # m=1, a=0, b=1
        self.assertEqual(center_text('abc', 4), 'abc ')


class TestInjectArg(unittest.TestCase):
    def test_adds_new_arg(self):
        result = inject_arg('--foo 1', '--bar', '2')
        self.assertEqual(result, '--foo 1 --bar 2')

    def test_overrides_spaced_arg(self):
        result = inject_arg('--foo 1 --bar 2', '--bar', '99')
        self.assertEqual(result, '--foo 1 --bar 99')

    def test_overrides_equals_arg(self):
        result = inject_arg('--foo=1 --bar=2', '--bar', '99')
        self.assertEqual(result, '--foo=1 --bar=99')

    def test_add_to_empty_cmdline(self):
        result = inject_arg('', '--foo', 'x')
        self.assertEqual(result, ' --foo x')

    def test_only_changes_matching_arg(self):
        result = inject_arg('--foobar 1 --foo 2', '--foo', '9')
        self.assertEqual(result, '--foobar 1 --foo 9')


class TestInjectArgs(unittest.TestCase):
    def test_multiple_overrides(self):
        result = inject_args('--a 1 --b 2', {'--a': '10', '--b': '20'})
        self.assertEqual(result, '--a 10 --b 20')

    def test_mix_override_and_add(self):
        result = inject_args('--a 1', {'--a': '10', '--b': '2'})
        self.assertEqual(result, '--a 10 --b 2')

    def test_empty_dict(self):
        self.assertEqual(inject_args('--a 1', {}), '--a 1')


class TestMakeArgsStr(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(make_args_str({'--foo': '1', '--bar': '2'}), '--foo 1 --bar 2')

    def test_none_value_is_bare_flag(self):
        self.assertEqual(make_args_str({'--verbose': None}), '--verbose')

    def test_mixed_flag_and_value(self):
        result = make_args_str({'--name': 'alice', '--dry-run': None})
        self.assertEqual(result, '--name alice --dry-run')

    def test_empty_dict(self):
        self.assertEqual(make_args_str({}), '')


if __name__ == '__main__':
    unittest.main()
