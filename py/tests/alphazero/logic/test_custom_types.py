import logging
import unittest
from alphazero.logic.custom_types import (
    ClientRole,
    Domain,
    GpuId,
    Version,
)


class TestVersion(unittest.TestCase):
    def test_str(self):
        self.assertEqual(str(Version(3)), 'v3')

    def test_str_zero(self):
        self.assertEqual(str(Version(0)), 'v0')

    def test_repr(self):
        self.assertEqual(repr(Version(7)), 'v7')

    def test_equality(self):
        self.assertEqual(Version(5), Version(5))
        self.assertNotEqual(Version(5), Version(6))


class TestClientRole(unittest.TestCase):
    def test_worker_roles_contains_self_play_worker(self):
        self.assertIn(ClientRole.SELF_PLAY_WORKER, ClientRole.worker_roles())

    def test_worker_roles_contains_ratings_worker(self):
        self.assertIn(ClientRole.RATINGS_WORKER, ClientRole.worker_roles())

    def test_worker_roles_does_not_contain_servers(self):
        roles = ClientRole.worker_roles()
        self.assertNotIn(ClientRole.SELF_PLAY_SERVER, roles)
        self.assertNotIn(ClientRole.RATINGS_SERVER, roles)

    def test_connection_log_level_worker_is_debug(self):
        self.assertEqual(
            ClientRole.connection_log_level(ClientRole.RATINGS_WORKER), logging.DEBUG
        )

    def test_connection_log_level_server_is_info(self):
        self.assertEqual(
            ClientRole.connection_log_level(ClientRole.SELF_PLAY_SERVER), logging.INFO
        )


class TestDomain(unittest.TestCase):
    def test_from_role_self_play_server(self):
        self.assertEqual(Domain.from_role(ClientRole.SELF_PLAY_SERVER), Domain.SELF_PLAY)

    def test_from_role_self_play_worker(self):
        self.assertEqual(Domain.from_role(ClientRole.SELF_PLAY_WORKER), Domain.SELF_PLAY)

    def test_from_role_ratings_server(self):
        self.assertEqual(Domain.from_role(ClientRole.RATINGS_SERVER), Domain.RATINGS)

    def test_from_role_ratings_worker(self):
        self.assertEqual(Domain.from_role(ClientRole.RATINGS_WORKER), Domain.RATINGS)

    def test_from_role_eval_vs_benchmark_server(self):
        self.assertEqual(
            Domain.from_role(ClientRole.EVAL_VS_BENCHMARK_SERVER), Domain.EVAL_VS_BENCHMARK
        )

    def test_from_role_eval_vs_benchmark_worker(self):
        self.assertEqual(
            Domain.from_role(ClientRole.EVAL_VS_BENCHMARK_WORKER), Domain.EVAL_VS_BENCHMARK
        )

    def test_from_role_invalid_raises(self):
        with self.assertRaises(ValueError):
            Domain.from_role(ClientRole.SELF_PLAY_SERVER.__class__('bad-role'))  # type: ignore

    def test_others_excludes_self(self):
        others = Domain.others(Domain.SELF_PLAY)
        self.assertNotIn(Domain.SELF_PLAY, others)

    def test_others_contains_remaining(self):
        others = Domain.others(Domain.SELF_PLAY)
        self.assertIn(Domain.TRAINING, others)
        self.assertIn(Domain.RATINGS, others)

    def test_others_length(self):
        # All domains minus 1
        all_domains = list(Domain)
        self.assertEqual(len(Domain.others(Domain.TRAINING)), len(all_domains) - 1)


class TestGpuId(unittest.TestCase):
    def test_str(self):
        gpu = GpuId(ip_address='192.168.1.1', device='cuda:0')
        self.assertEqual(str(gpu), 'Gpu(cuda:0@192.168.1.1)')

    def test_equality(self):
        a = GpuId('10.0.0.1', 'cuda:0')
        b = GpuId('10.0.0.1', 'cuda:0')
        self.assertEqual(a, b)

    def test_inequality_different_device(self):
        a = GpuId('10.0.0.1', 'cuda:0')
        b = GpuId('10.0.0.1', 'cuda:1')
        self.assertNotEqual(a, b)

    def test_inequality_different_ip(self):
        a = GpuId('10.0.0.1', 'cuda:0')
        b = GpuId('10.0.0.2', 'cuda:0')
        self.assertNotEqual(a, b)

    def test_hashable(self):
        gpu = GpuId('10.0.0.1', 'cuda:0')
        s = {gpu}
        self.assertIn(gpu, s)


if __name__ == '__main__':
    unittest.main()
