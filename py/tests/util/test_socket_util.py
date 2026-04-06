from util.socket_util import encode_json, EncodedJson

import json
import struct
import unittest


class TestEncodeJson(unittest.TestCase):

    def test_simple_dict(self):
        data = {'key': 'value'}
        result = encode_json(data)
        self.assertIsInstance(result, EncodedJson)

        # Verify header is 4-byte big-endian length
        payload_len = struct.unpack('>I', result.header)[0]
        self.assertEqual(payload_len, len(result.payload))

        # Verify payload decodes to the original data
        decoded = json.loads(result.payload.decode())
        self.assertEqual(decoded, data)

    def test_empty_dict(self):
        data = {}
        result = encode_json(data)
        payload_len = struct.unpack('>I', result.header)[0]
        self.assertEqual(payload_len, len(result.payload))
        decoded = json.loads(result.payload.decode())
        self.assertEqual(decoded, {})

    def test_nested_dict(self):
        data = {'outer': {'inner': [1, 2, 3]}}
        result = encode_json(data)
        decoded = json.loads(result.payload.decode())
        self.assertEqual(decoded, data)

    def test_dict_with_numbers(self):
        data = {'int': 42, 'float': 3.14, 'negative': -1}
        result = encode_json(data)
        decoded = json.loads(result.payload.decode())
        self.assertEqual(decoded, data)

    def test_dict_with_null(self):
        data = {'key': None}
        result = encode_json(data)
        decoded = json.loads(result.payload.decode())
        self.assertEqual(decoded, data)

    def test_header_is_4_bytes(self):
        result = encode_json({'a': 1})
        self.assertEqual(len(result.header), 4)

    def test_large_payload(self):
        data = {f'key_{i}': f'value_{i}' for i in range(1000)}
        result = encode_json(data)
        payload_len = struct.unpack('>I', result.header)[0]
        self.assertEqual(payload_len, len(result.payload))
        decoded = json.loads(result.payload.decode())
        self.assertEqual(decoded, data)


if __name__ == '__main__':
    unittest.main()
