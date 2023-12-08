from util.infinite_sequence import InfiniteSequence
import unittest


def trim(s: str) -> str:
    # remove leading and trailing whitespace from each line
    lines = s.split('\n')
    lines = [line.strip() for line in lines]
    # remove empty lines
    lines = [line for line in lines if line]
    return '\n'.join(lines)


class TestInfiniteSequence(unittest.TestCase):
    def custom_str_assert(self, S, expected):
        self.assertEqual(S.to_string('\n'), trim(expected))

    def test_all(self):
        S = InfiniteSequence(value_fmt='%.f')
        S.check_invariants()
        self.custom_str_assert(S, '[0, inf): 0')
        self.assertEqual(S[0], 0)
        self.assertEqual(S[999999], 0)
        self.assertEqual(S.sum(), 0)
        self.assertEqual(S[:50].sum(), 0)
        self.assertEqual(S[50:150].sum(), 0)
        self.assertEqual(S[150:].sum(), 0)

        S[100] = 1
        S.check_invariants()
        self.custom_str_assert(S,
                               '''
                               [0, 100): 0
                               [100, 101): 1
                               [101, inf): 0
                               '''
                               )
        self.assertEqual(S[99], 0)
        self.assertEqual(S[100], 1)
        self.assertEqual(S[101], 0)
        self.assertEqual(S[999999], 0)
        self.assertEqual(S.sum(), 1)
        self.assertEqual(S[:50].sum(), 0)
        self.assertEqual(S[50:150].sum(), 1)
        self.assertEqual(S[150:].sum(), 0)

        S[:400] += 2
        S.check_invariants()
        self.custom_str_assert(S,
                               '''
                               [0, 100): 2
                               [100, 101): 3
                               [101, 400): 2
                               [400, inf): 0
                               '''
                               )
        self.assertEqual(S[99], 2)
        self.assertEqual(S[100], 3)
        self.assertEqual(S[101], 2)
        self.assertEqual(S[399], 2)
        self.assertEqual(S[400], 0)
        self.assertEqual(S[999999], 0)
        self.assertEqual(S.sum(), 801)
        self.assertEqual(S[:50].sum(), 100)
        self.assertEqual(S[50:150].sum(), 201)
        self.assertEqual(S[150:].sum(), 500)

        S[300:400] = 0
        S.check_invariants()
        self.custom_str_assert(S,
                               '''
                               [0, 100): 2
                               [100, 101): 3
                               [101, 300): 2
                               [300, inf): 0
                               '''
                               )
        self.assertEqual(S[99], 2)
        self.assertEqual(S[100], 3)
        self.assertEqual(S[101], 2)
        self.assertEqual(S[299], 2)
        self.assertEqual(S[300], 0)
        self.assertEqual(S[399], 0)
        self.assertEqual(S[400], 0)
        self.assertEqual(S[999999], 0)
        self.assertEqual(S.sum(), 601)
        self.assertEqual(S[:50].sum(), 100)
        self.assertEqual(S[50:150].sum(), 201)
        self.assertEqual(S[150:].sum(), 300)

        S[101:150] += 1
        S.check_invariants()
        self.custom_str_assert(S,
                               '''
                               [0, 100): 2
                               [100, 150): 3
                               [150, 300): 2
                               [300, inf): 0
                               '''
                               )
        self.assertEqual(S[99], 2)
        self.assertEqual(S[100], 3)
        self.assertEqual(S[101], 3)
        self.assertEqual(S[149], 3)
        self.assertEqual(S[150], 2)
        self.assertEqual(S[151], 2)
        self.assertEqual(S[299], 2)
        self.assertEqual(S[300], 0)
        self.assertEqual(S[999999], 0)
        self.assertEqual(S.sum(), 650)
        self.assertEqual(S[:50].sum(), 100)
        self.assertEqual(S[50:150].sum(), 250)
        self.assertEqual(S[150:].sum(), 300)

        S[150:] -= -1
        S.check_invariants()
        self.custom_str_assert(S,
                               '''
                               [0, 100): 2
                               [100, 300): 3
                               [300, inf): 1
                               '''
                               )
        self.assertEqual(S[99], 2)
        self.assertEqual(S[100], 3)
        self.assertEqual(S[101], 3)
        self.assertEqual(S[149], 3)
        self.assertEqual(S[150], 3)
        self.assertEqual(S[151], 3)
        self.assertEqual(S[299], 3)
        self.assertEqual(S[300], 1)
        self.assertEqual(S[999999], 1)
        self.assertEqual(S.sum(), float('inf'))
        self.assertEqual(S[:50].sum(), 100)
        self.assertEqual(S[50:150].sum(), 250)
        self.assertEqual(S[150:].sum(), float('inf'))


if __name__ == '__main__':
    unittest.main()
