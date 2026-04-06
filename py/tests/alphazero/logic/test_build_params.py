from alphazero.logic.build_params import BuildParams

import unittest


class TestBuildParamsPostInit(unittest.TestCase):

    def test_default_construction(self):
        bp = BuildParams()
        self.assertIsNone(bp.debug_build)
        self.assertIsNone(bp.binary_path)
        self.assertFalse(bp.override_binary)
        self.assertFalse(bp.use_stored_binary)

    def test_binary_path_with_debug_build_raises(self):
        with self.assertRaises(ValueError):
            BuildParams(binary_path='/some/path', debug_build=True)

    def test_binary_path_with_override_binary_raises(self):
        with self.assertRaises(ValueError):
            BuildParams(binary_path='/some/path', override_binary=True)

    def test_binary_path_with_use_stored_binary_raises(self):
        with self.assertRaises(ValueError):
            BuildParams(binary_path='/some/path', use_stored_binary=True)

    def test_debug_build_with_override_raises(self):
        with self.assertRaises(ValueError):
            BuildParams(debug_build=True, override_binary=True)

    def test_debug_build_with_use_stored_raises(self):
        with self.assertRaises(ValueError):
            BuildParams(debug_build=True, use_stored_binary=True)

    def test_binary_path_alone_ok(self):
        bp = BuildParams(binary_path='/my/binary')
        self.assertEqual(bp.binary_path, '/my/binary')

    def test_debug_build_alone_ok(self):
        bp = BuildParams(debug_build=True)
        self.assertTrue(bp.debug_build)

    def test_use_stored_binary_alone_ok(self):
        bp = BuildParams(use_stored_binary=True)
        self.assertTrue(bp.use_stored_binary)


class TestBuildType(unittest.TestCase):

    def test_debug(self):
        bp = BuildParams(debug_build=True)
        self.assertEqual(bp.build_type, 'Debug')

    def test_release(self):
        bp = BuildParams()
        self.assertEqual(bp.build_type, 'Release')

    def test_release_explicit_false(self):
        bp = BuildParams(debug_build=False)
        self.assertEqual(bp.build_type, 'Release')


class TestGetBinaryPath(unittest.TestCase):

    def test_default_release(self):
        bp = BuildParams()
        self.assertEqual(bp.get_binary_path('c4'), 'target/Release/bin/c4')

    def test_debug(self):
        bp = BuildParams(debug_build=True)
        self.assertEqual(bp.get_binary_path('c4'), 'target/Debug/bin/c4')

    def test_custom_binary_path(self):
        bp = BuildParams(binary_path='/custom/bin/c4')
        self.assertEqual(bp.get_binary_path('c4'), '/custom/bin/c4')
        # game arg is ignored when binary_path is set
        self.assertEqual(bp.get_binary_path('tictactoe'), '/custom/bin/c4')


class TestGetFfiLibPath(unittest.TestCase):

    def test_default_release(self):
        bp = BuildParams()
        self.assertEqual(bp.get_ffi_lib_path('c4'), 'target/Release/lib/libc4.so')

    def test_debug(self):
        bp = BuildParams(debug_build=True)
        self.assertEqual(bp.get_ffi_lib_path('c4'), 'target/Debug/lib/libc4.so')

    def test_custom_ffi_lib_path(self):
        bp = BuildParams(ffi_lib_path='/custom/lib/libc4.so')
        self.assertEqual(bp.get_ffi_lib_path('c4'), '/custom/lib/libc4.so')


class TestAddToCmd(unittest.TestCase):

    def test_defaults_empty(self):
        bp = BuildParams()
        cmd = []
        bp.add_to_cmd(cmd, loop_controller=True)
        self.assertEqual(cmd, [])

    def test_binary_path(self):
        bp = BuildParams(binary_path='/my/bin')
        cmd = []
        bp.add_to_cmd(cmd)
        self.assertEqual(cmd, ['--binary-path', '/my/bin'])

    def test_debug_build(self):
        bp = BuildParams(debug_build=True)
        cmd = []
        bp.add_to_cmd(cmd, loop_controller=True)
        self.assertIn('--debug-build', cmd)

    def test_override_binary(self):
        bp = BuildParams(override_binary=True)
        cmd = []
        bp.add_to_cmd(cmd, loop_controller=True)
        self.assertIn('--override-binary', cmd)

    def test_use_stored_binary(self):
        bp = BuildParams(use_stored_binary=True)
        cmd = []
        bp.add_to_cmd(cmd, loop_controller=True)
        self.assertIn('--use-stored-binary', cmd)

    def test_non_loop_controller_skips_debug_flags(self):
        bp = BuildParams(debug_build=True)
        cmd = []
        bp.add_to_cmd(cmd, loop_controller=False)
        self.assertNotIn('--debug-build', cmd)


if __name__ == '__main__':
    unittest.main()
