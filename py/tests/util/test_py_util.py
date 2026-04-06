from util.py_util import (
    is_iterable,
    is_valid_path_component,
    make_hidden_filename,
    find_largest_gap,
    CustomHelpFormatter,
)

import argparse
import unittest


class TestIsIterable(unittest.TestCase):

    def test_list(self):
        self.assertTrue(is_iterable([1, 2, 3]))

    def test_string(self):
        self.assertTrue(is_iterable('hello'))

    def test_tuple(self):
        self.assertTrue(is_iterable((1, 2)))

    def test_set(self):
        self.assertTrue(is_iterable({1, 2}))

    def test_dict(self):
        self.assertTrue(is_iterable({'a': 1}))

    def test_generator(self):
        self.assertTrue(is_iterable(x for x in range(3)))

    def test_int(self):
        self.assertFalse(is_iterable(42))

    def test_none(self):
        self.assertFalse(is_iterable(None))

    def test_float(self):
        self.assertFalse(is_iterable(3.14))

    def test_bool(self):
        self.assertFalse(is_iterable(True))

    def test_empty_list(self):
        self.assertTrue(is_iterable([]))


class TestIsValidPathComponent(unittest.TestCase):

    def test_simple_name(self):
        self.assertTrue(is_valid_path_component('myfile'))

    def test_name_with_extension(self):
        self.assertTrue(is_valid_path_component('myfile.txt'))

    def test_name_with_hyphen(self):
        self.assertTrue(is_valid_path_component('my-file'))

    def test_parent_dir(self):
        # '..' is a valid path component per os.path.split
        self.assertTrue(is_valid_path_component('..'))

    def test_path_traversal(self):
        self.assertFalse(is_valid_path_component('../etc'))

    def test_slash(self):
        self.assertFalse(is_valid_path_component('a/b'))

    def test_empty_string(self):
        # '' is treated as valid by os.path.split (returns '')
        self.assertTrue(is_valid_path_component(''))

    def test_current_dir(self):
        # '.' is a valid path component per os.path.split
        self.assertTrue(is_valid_path_component('.'))


class TestMakeHiddenFilename(unittest.TestCase):

    def test_simple_file(self):
        self.assertEqual(make_hidden_filename('foo.txt'), '.foo.txt')

    def test_file_in_directory(self):
        self.assertEqual(make_hidden_filename('dir/foo.txt'), 'dir/.foo.txt')

    def test_nested_directory(self):
        self.assertEqual(make_hidden_filename('a/b/c.txt'), 'a/b/.c.txt')

    def test_no_extension(self):
        self.assertEqual(make_hidden_filename('myfile'), '.myfile')

    def test_already_hidden(self):
        self.assertEqual(make_hidden_filename('.hidden'), '..hidden')


class TestFindLargestGap(unittest.TestCase):

    def test_basic(self):
        self.assertEqual(find_largest_gap([1, 2, 5, 6]), (2, 5))

    def test_two_elements(self):
        self.assertEqual(find_largest_gap([10, 20]), (10, 20))

    def test_equal_gaps(self):
        # When gaps are equal, returns first occurrence
        result = find_largest_gap([1, 3, 5, 7])
        self.assertEqual(result, (1, 3))

    def test_gap_at_end(self):
        self.assertEqual(find_largest_gap([1, 2, 3, 100]), (3, 100))

    def test_gap_at_beginning(self):
        self.assertEqual(find_largest_gap([0, 100, 101, 102]), (0, 100))

    def test_negative_numbers(self):
        self.assertEqual(find_largest_gap([-10, -5, 0, 1]), (-10, -5))

    def test_floats(self):
        self.assertEqual(find_largest_gap([0.1, 0.2, 0.9]), (0.2, 0.9))


class TestCustomHelpFormatter(unittest.TestCase):

    def test_format(self):
        parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
        parser.add_argument('-t', '--tag', help='tag for this run')
        help_text = parser.format_help()
        self.assertIn('-t/--tag', help_text)
        # Should NOT have the default format with comma
        self.assertNotIn('-t TAG, --tag TAG', help_text)


if __name__ == '__main__':
    unittest.main()
