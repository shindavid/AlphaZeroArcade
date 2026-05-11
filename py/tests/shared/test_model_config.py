from shared.model_config import ModelConfig, ModuleSpec, ModuleSequenceSpec

import unittest


class TestModelConfigCreate(unittest.TestCase):

    def test_simple_valid_dag(self):
        config = ModelConfig.create(
            trunk=ModuleSpec('ConvBlock', args=[2, 32, 3]),
            head=ModuleSpec('PolicyHead', args=[32, 9], parents=['trunk']),
        )
        self.assertIn('trunk', config.parts)
        self.assertIn('head', config.parts)

    def test_multi_head(self):
        config = ModelConfig.create(
            trunk=ModuleSpec('ConvBlock', args=[2, 32, 3]),
            policy=ModuleSpec('PolicyHead', args=[32, 9], parents=['trunk']),
            value=ModuleSpec('WinLossValueHead', args=[32, 1], parents=['trunk']),
        )
        self.assertEqual(len(config.parts), 3)


class TestModelConfigTrim(unittest.TestCase):

    def test_trim_keeps_ancestors(self):
        config = ModelConfig.create(
            trunk=ModuleSpec('ConvBlock', args=[2, 32, 3]),
            policy=ModuleSpec('PolicyHead', args=[32, 9], parents=['trunk']),
            value=ModuleSpec('WinLossValueHead', args=[32, 1], parents=['trunk']),
        )
        trimmed = config.trim({'policy'})
        self.assertIn('trunk', trimmed.parts)
        self.assertIn('policy', trimmed.parts)
        self.assertNotIn('value', trimmed.parts)

    def test_trim_keeps_deep_ancestors(self):
        config = ModelConfig.create(
            input_block=ModuleSpec('ConvBlock', args=[2, 32, 3]),
            res=ModuleSpec('ResBlock', args=[32], parents=['input_block']),
            head=ModuleSpec('PolicyHead', args=[32, 9], parents=['res']),
        )
        trimmed = config.trim({'head'})
        self.assertEqual(set(trimmed.parts.keys()), {'input_block', 'res', 'head'})

    def test_trim_unknown_part_raises(self):
        config = ModelConfig.create(
            trunk=ModuleSpec('ConvBlock', args=[2, 32, 3]),
            head=ModuleSpec('PolicyHead', args=[32, 9], parents=['trunk']),
        )
        with self.assertRaises(AssertionError):
            config.trim({'nonexistent'})

    def test_trim_empty_raises(self):
        config = ModelConfig.create(
            trunk=ModuleSpec('ConvBlock', args=[2, 32, 3]),
            head=ModuleSpec('PolicyHead', args=[32, 9], parents=['trunk']),
        )
        with self.assertRaises(AssertionError):
            config.trim(set())


class TestModelConfigValidation(unittest.TestCase):

    def test_unknown_module_type_raises(self):
        with self.assertRaises(AssertionError):
            ModelConfig.create(
                trunk=ModuleSpec('NonExistentModule', args=[]),
            )

    def test_undeclared_external_input_raises(self):
        # A parent name that is neither a module key nor declared in `external_inputs` is an
        # error: the framework no longer auto-infers external inputs from missing parents.
        with self.assertRaises(AssertionError):
            ModelConfig.create(
                head=ModuleSpec('PolicyHead', args=[32, 9], parents=['missing_trunk']),
            )

    def test_declared_external_input_accepted(self):
        # Declaring an extra external input makes it a valid parent reference.
        config = ModelConfig.create(
            external_inputs=['extra'],
            stem=ModuleSpec('ConvBlock', args=[2, 32, 3]),
            head=ModuleSpec('PolicyHead', args=[32, 9], parents=['stem', 'extra']),
        )
        self.assertEqual(set(config.external_inputs), {'extra'})

    def test_unreferenced_external_input_raises(self):
        with self.assertRaises(AssertionError):
            ModelConfig.create(
                external_inputs=['unused'],
                trunk=ModuleSpec('ConvBlock', args=[2, 32, 3]),
                head=ModuleSpec('PolicyHead', args=[32, 9], parents=['trunk']),
            )

    def test_external_input_name_clashes_with_module_raises(self):
        with self.assertRaises(AssertionError):
            ModelConfig.create(
                external_inputs=['trunk'],
                trunk=ModuleSpec('ConvBlock', args=[2, 32, 3]),
                head=ModuleSpec('PolicyHead', args=[32, 9], parents=['trunk']),
            )

    def test_no_input_module_raises(self):
        # All modules have parents → no input
        with self.assertRaises(AssertionError):
            ModelConfig.create(
                a=ModuleSpec('ConvBlock', args=[2, 32, 3], parents=['b']),
                b=ModuleSpec('PolicyHead', args=[32, 9], parents=['a']),
            )

    def test_non_head_leaf_allowed(self):
        # Non-Head leaves are allowed: they're training-only modules (e.g. BetaZero's BackupNet)
        # whose outputs are consumed by loss terms outside the inference graph.
        config = ModelConfig.create(
            trunk=ModuleSpec('ConvBlock', args=[2, 32, 3]),
        )
        self.assertEqual(set(config.parts.keys()), {'trunk'})

    def test_module_sequence_spec(self):
        config = ModelConfig.create(
            trunk=ModuleSequenceSpec(
                ModuleSpec('ConvBlock', args=[2, 32, 3]),
                ModuleSpec('ResBlock', args=[32]),
            ),
            head=ModuleSpec('PolicyHead', args=[32, 9], parents=['trunk']),
        )
        self.assertIn('trunk', config.parts)


if __name__ == '__main__':
    unittest.main()
