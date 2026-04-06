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

    def test_missing_parent_raises(self):
        with self.assertRaises(AssertionError):
            ModelConfig.create(
                head=ModuleSpec('PolicyHead', args=[32, 9], parents=['missing_trunk']),
            )

    def test_no_input_module_raises(self):
        # All modules have parents → no input
        with self.assertRaises(AssertionError):
            ModelConfig.create(
                a=ModuleSpec('ConvBlock', args=[2, 32, 3], parents=['b']),
                b=ModuleSpec('PolicyHead', args=[32, 9], parents=['a']),
            )

    def test_non_head_leaf_raises(self):
        # ConvBlock is not a Head subclass, so it can't be a leaf
        with self.assertRaises(Exception):
            ModelConfig.create(
                trunk=ModuleSpec('ConvBlock', args=[2, 32, 3]),
                # 'trunk' is a leaf but not a Head → should fail
            )

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
