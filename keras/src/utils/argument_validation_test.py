import numpy as np

from keras.src.testing import test_case
from keras.src.utils import argument_validation


class StandardizeTupleTest(test_case.TestCase):
    def test_int(self):
        self.assertEqual(
            argument_validation.standardize_tuple(3, 2, "kernel_size"), (3, 3)
        )

    def test_numpy_scalar_int(self):
        result = argument_validation.standardize_tuple(
            np.int64(2), 2, "kernel_size"
        )
        self.assertEqual(result, (2, 2))
        self.assertTrue(all(type(v) is int for v in result))

    def test_iterable_values_are_converted_to_int(self):
        for value in [(2.0, 3.0), [np.int32(2), np.int64(3)], np.array([2, 3])]:
            result = argument_validation.standardize_tuple(
                value, 2, "kernel_size"
            )
            self.assertEqual(result, (2, 3))
            self.assertTrue(all(type(v) is int for v in result))

    def test_allow_zero(self):
        self.assertEqual(
            argument_validation.standardize_tuple(
                np.int64(0), 2, "padding", allow_zero=True
            ),
            (0, 0),
        )
        with self.assertRaisesRegex(ValueError, "do not satisfy"):
            argument_validation.standardize_tuple(np.int64(0), 2, "strides")

    def test_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "must be a tuple of 2"):
            argument_validation.standardize_tuple((1, 2, 3), 2, "strides")
        with self.assertRaisesRegex(ValueError, "must be a tuple of 2"):
            argument_validation.standardize_tuple(None, 2, "strides")
        with self.assertRaisesRegex(ValueError, "do not satisfy"):
            argument_validation.standardize_tuple(-1, 2, "strides")
        with self.assertRaisesRegex(ValueError, "type"):
            argument_validation.standardize_tuple(("a", 1), 2, "strides")
