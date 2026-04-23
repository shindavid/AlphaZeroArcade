from alphazero.logic.agent_types import (
    AgentRole,
    MCTSAgent,
    IndexedAgent,
    MatchType,
    ReferenceAgent,
)

import unittest


class TestMCTSAgent(unittest.TestCase):

    def test_default_values(self):
        agent = MCTSAgent()
        self.assertEqual(agent.spec_name, 'alpha0')
        self.assertEqual(agent.gen, 0)
        self.assertIsNone(agent.n_iters)
        self.assertFalse(agent.set_temp_zero)
        self.assertIsNone(agent.tag)
        self.assertIsNone(agent.binary)
        self.assertIsNone(agent.model)
        self.assertEqual(agent.extra_player_args, ())
        self.assertEqual(agent.extra_file_args, frozenset())

    def test_name_property(self):
        agent = MCTSAgent(spec_name='alpha0', gen=5)
        self.assertEqual(agent.name, 'alpha0-5')

    def test_level_property(self):
        agent = MCTSAgent(gen=10)
        self.assertEqual(agent.level, 10)

    def test_to_dict(self):
        agent = MCTSAgent(spec_name='alpha0', gen=3, n_iters=100)
        d = agent.to_dict()
        self.assertEqual(d['type'], 'MCTS')
        self.assertEqual(d['data']['spec_name'], 'alpha0')
        self.assertEqual(d['data']['gen'], 3)
        self.assertEqual(d['data']['n_iters'], 100)

    def test_make_player_str_gen0(self):
        agent = MCTSAgent(gen=0)
        s = agent.make_player_str('/run/dir')
        self.assertIn('--type alpha0-C', s)
        self.assertIn('--no-model', s)
        self.assertIn('--name alpha0-0', s)

    def test_make_player_str_with_model(self):
        agent = MCTSAgent(gen=5, model='models/gen-5.onnx')
        s = agent.make_player_str('/run/dir')
        self.assertIn('-m /run/dir/models/gen-5.onnx', s)
        self.assertNotIn('--no-model', s)

    def test_make_player_str_with_n_iters(self):
        agent = MCTSAgent(gen=1, n_iters=200, model='models/gen-1.onnx')
        s = agent.make_player_str('/run/dir')
        self.assertIn('-i 200', s)
        self.assertIn('--name alpha0-1-200', s)

    def test_make_player_str_with_temp_zero(self):
        agent = MCTSAgent(gen=0, set_temp_zero=True)
        s = agent.make_player_str('/run/dir')
        self.assertIn('--starting-move-temp 0', s)
        self.assertIn('--ending-move-temp 0', s)

    def test_make_player_str_with_suffix(self):
        agent = MCTSAgent(gen=0)
        s = agent.make_player_str('/run/dir', suffix='_test')
        self.assertIn('--name alpha0-0_test', s)

    def test_frozen(self):
        agent = MCTSAgent()
        with self.assertRaises(AttributeError):
            agent.gen = 5

    def test_str(self):
        agent = MCTSAgent(gen=3)
        self.assertEqual(str(agent), 'MCTSAgent-gen-3')

    def test_extra_player_args_non_file(self):
        agent = MCTSAgent(gen=0,
                          extra_player_args=(('--foo', 'bar'),))
        s = agent.make_player_str('/run/dir')
        self.assertIn('--foo bar', s)

    def test_extra_player_args_file(self):
        agent = MCTSAgent(gen=5, model='models/gen-5.onnx',
                          extra_player_args=(('--backup-nn-model', 'aux/gen-5.bin'),),
                          extra_file_args=frozenset({'--backup-nn-model'}))
        s = agent.make_player_str('/run/dir')
        self.assertIn('--backup-nn-model /run/dir/aux/gen-5.bin', s)

    def test_extra_file_args_excluded_from_equality(self):
        # extra_file_args has compare=False, so agents differing only in extra_file_args are equal
        a1 = MCTSAgent(gen=5, extra_file_args=frozenset({'--backup-nn-model'}))
        a2 = MCTSAgent(gen=5, extra_file_args=frozenset())
        self.assertEqual(a1, a2)
        self.assertEqual(hash(a1), hash(a2))


class TestMCTSAgentBeta0(unittest.TestCase):
    """Tests for MCTSAgent used with a beta0 spec_name."""

    def test_default_values(self):
        agent = MCTSAgent(spec_name='beta0')
        self.assertEqual(agent.spec_name, 'beta0')
        self.assertEqual(agent.gen, 0)
        self.assertIsNone(agent.n_iters)
        self.assertFalse(agent.set_temp_zero)

    def test_to_dict_includes_extra_fields(self):
        agent = MCTSAgent(spec_name='beta0', gen=3, n_iters=100)
        d = agent.to_dict()
        self.assertEqual(d['type'], 'MCTS')
        self.assertEqual(d['data']['spec_name'], 'beta0')
        self.assertIn('extra_player_args', d['data'])
        self.assertIn('extra_file_args', d['data'])

    def test_make_player_str_gen0_no_aux(self):
        agent = MCTSAgent(spec_name='beta0', gen=0)
        s = agent.make_player_str('/run/dir')
        self.assertIn('--type beta0-C', s)
        self.assertIn('--no-model', s)
        self.assertNotIn('--backup-nn-model', s)

    def test_make_player_str_with_aux_model(self):
        agent = MCTSAgent(spec_name='beta0', gen=5, model='models/gen-5.onnx',
                          extra_player_args=(('--backup-nn-model', 'aux/gen-5.bin'),),
                          extra_file_args=frozenset({'--backup-nn-model'}))
        s = agent.make_player_str('/run/dir')
        self.assertIn('-m /run/dir/models/gen-5.onnx', s)
        self.assertIn('--backup-nn-model /run/dir/aux/gen-5.bin', s)

    def test_str(self):
        agent = MCTSAgent(spec_name='beta0', gen=4)
        self.assertEqual(str(agent), 'MCTSAgent-gen-4')

    def test_frozen(self):
        agent = MCTSAgent(spec_name='beta0')
        with self.assertRaises(AttributeError):
            agent.gen = 5

    def test_level_property(self):
        agent = MCTSAgent(spec_name='beta0', gen=8)
        self.assertEqual(agent.level, 8)


class TestReferenceAgent(unittest.TestCase):

    def test_creation(self):
        agent = ReferenceAgent(type_str='Perfect', strength_param='--strength', strength=5)
        self.assertEqual(agent.type_str, 'Perfect')
        self.assertEqual(agent.strength, 5)

    def test_level_property(self):
        agent = ReferenceAgent(type_str='Perfect', strength_param='--strength', strength=10)
        self.assertEqual(agent.level, 10)

    def test_to_dict(self):
        agent = ReferenceAgent(type_str='Perfect', strength_param='--strength', strength=5)
        d = agent.to_dict()
        self.assertEqual(d['type'], 'Reference')
        self.assertEqual(d['data']['type_str'], 'Perfect')
        self.assertEqual(d['data']['strength'], 5)

    def test_make_player_str(self):
        agent = ReferenceAgent(type_str='Perfect', strength_param='--strength', strength=5)
        s = agent.make_player_str('/run/dir')
        self.assertIn('--type Perfect', s)
        self.assertIn('--name Perfect-5', s)
        self.assertIn('--strength 5', s)

    def test_make_player_str_with_suffix(self):
        agent = ReferenceAgent(type_str='Perfect', strength_param='--strength', strength=5)
        s = agent.make_player_str('/run/dir', suffix='_a')
        self.assertIn('--name Perfect-5_a', s)

    def test_frozen(self):
        agent = ReferenceAgent(type_str='Perfect', strength_param='--strength', strength=5)
        with self.assertRaises(AttributeError):
            agent.strength = 10


class TestAgentRole(unittest.TestCase):

    def test_to_str_single(self):
        s = AgentRole.to_str({AgentRole.BENCHMARK})
        self.assertEqual(s, 'benchmark')

    def test_to_str_multiple(self):
        s = AgentRole.to_str({AgentRole.BENCHMARK, AgentRole.TEST})
        # sorted alphabetically by value
        self.assertEqual(s, 'benchmark,test')

    def test_from_str_single(self):
        roles = AgentRole.from_str('benchmark')
        self.assertEqual(roles, {AgentRole.BENCHMARK})

    def test_from_str_multiple(self):
        roles = AgentRole.from_str('benchmark,test')
        self.assertEqual(roles, {AgentRole.BENCHMARK, AgentRole.TEST})

    def test_roundtrip(self):
        original = {AgentRole.BENCHMARK, AgentRole.TEST}
        s = AgentRole.to_str(original)
        recovered = AgentRole.from_str(s)
        self.assertEqual(original, recovered)


class TestIndexedAgent(unittest.TestCase):

    def test_to_dict(self):
        agent = MCTSAgent(gen=3)
        indexed = IndexedAgent(
            agent=agent,
            index=0,
            roles={AgentRole.TEST},
            db_id=42,
        )
        d = indexed.to_dict()
        self.assertEqual(d['index'], 0)
        self.assertEqual(d['db_id'], 42)
        self.assertEqual(d['roles'], 'test')
        self.assertEqual(d['agent']['type'], 'MCTS')


class TestMatchType(unittest.TestCase):

    def test_values(self):
        self.assertEqual(MatchType.BENCHMARK.value, 'benchmark')
        self.assertEqual(MatchType.EVALUATE.value, 'evaluate')


if __name__ == '__main__':
    unittest.main()
