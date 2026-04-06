from alphazero.logic.benchmark_record import BenchmarkRecord

import unittest


class TestBenchmarkRecord(unittest.TestCase):

    def _make_record(self):
        from alphazero.servers.loop_control.base_dir import VERSION
        return BenchmarkRecord(
            utc_key='2024-01-15_12-00-00.000000_UTC',
            tag='v1',
            game='c4',
            version=VERSION,
        )

    def test_to_dict(self):
        record = self._make_record()
        d = record.to_dict()
        self.assertEqual(d['utc_key'], '2024-01-15_12-00-00.000000_UTC')
        self.assertEqual(d['tag'], 'v1')
        self.assertIn('version', d)

    def test_to_dict_keys(self):
        record = self._make_record()
        d = record.to_dict()
        self.assertEqual(set(d.keys()), {'utc_key', 'tag', 'version'})

    def test_key(self):
        record = self._make_record()
        key = record.key()
        self.assertIn('c4', key)
        self.assertIn('v1', key)
        self.assertIn('.tar', key)

    def test_docker_image_ref(self):
        record = self._make_record()
        ref = record.docker_image_ref()
        self.assertIn('dshin83/alphazeroarcade-benchmarks:', ref)
        self.assertIn('c4', ref)
        self.assertIn('v1', ref)
        # Uses periods as delimiters
        parts = ref.split(':')[1].split('.')
        self.assertGreaterEqual(len(parts), 4)

    def test_version_matches(self):
        record = self._make_record()
        self.assertTrue(record.version_matches())

    def test_version_mismatch(self):
        from alphazero.logic.custom_types import Version
        record = BenchmarkRecord(
            utc_key='2024-01-15_12-00-00.000000_UTC',
            tag='v1',
            game='c4',
            version=Version(num=999),
        )
        self.assertFalse(record.version_matches())

    def test_default_record(self):
        record = BenchmarkRecord()
        self.assertIsNone(record.utc_key)
        self.assertIsNone(record.tag)
        self.assertIsNone(record.game)


if __name__ == '__main__':
    unittest.main()
