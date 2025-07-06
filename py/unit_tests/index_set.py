from util.index_set import IndexSet

import numpy as np
import unittest

class TestIndexSet(unittest.TestCase):

    def test_add_and_contains(self):
        s = IndexSet()
        s.add(2)
        s.add(5)
        self.assertIn(2, s)
        self.assertIn(5, s)
        self.assertNotIn(3, s)

    def test_len_and_iteration(self):
        s = IndexSet()
        s.add(1)
        s.add(3)
        s.add(7)
        self.assertEqual(len(s), 3)
        self.assertEqual(set(s), {1, 3, 7})

    def test_remove(self):
        s = IndexSet()
        s.add(4)
        s.add(6)
        s.remove(4)
        self.assertNotIn(4, s)
        self.assertIn(6, s)
        with self.assertRaises(KeyError):
            s.remove(4)

    def test_discard(self):
        s = IndexSet()
        s.add(9)
        s.discard(9)
        self.assertNotIn(9, s)
        s.discard(9)
        self.assertNotIn(9, s)

    def test_auto_resize(self):
        s = IndexSet()
        s.add(100)
        self.assertIn(100, s)
        self.assertNotIn(99, s)

    def test_repr(self):
        s = IndexSet()
        s.add(3)
        s.add(8)
        rep = repr(s)
        self.assertEqual(rep, 'IndexSet([np.int64(3), np.int64(8)])')

    def test_array_conversion(self):
        s = IndexSet()
        s.add(1)
        s.add(2)
        arr = np.array(s)
        self.assertEqual(arr.dtype, bool)
        self.assertTrue(arr[1])
        self.assertTrue(arr[2])
        self.assertFalse(arr[0])

    def test_add_negative(self):
        s = IndexSet()
        with self.assertRaises(IndexError):
            s.add(-1)

    def test_remove_out_of_range(self):
        s = IndexSet()
        with self.assertRaises(KeyError):
            s.remove(100)

    def test_subscript(self):
        s = IndexSet()
        s.add(1)
        s.add(3)
        self.assertTrue(s[1])
        self.assertTrue(s[3])
        self.assertFalse(s[0])

if __name__ == '__main__':
    unittest.main()
