import numpy as np
import pytest
import tree

from openpi_client import msgpack_numpy


def _check(expected, actual):
    if isinstance(expected, np.ndarray):
        assert expected.shape == actual.shape
        assert expected.dtype == actual.dtype
        assert np.array_equal(expected, actual, equal_nan=expected.dtype.kind == "f")
    else:
        assert expected == actual


@pytest.mark.parametrize(
    "data",
    [
        1,  # int
        1.0,  # float
        "hello",  # string
        np.bool_(True),  # boolean scalar
        np.array([1, 2, 3])[0],  # int scalar
        np.str_("asdf"),  # string scalar
        [1, 2, 3],  # list
        {"key": "value"},  # dict
        {"key": [1, 2, 3]},  # nested dict
        np.array(1.0),  # 0D array
        np.array([1, 2, 3], dtype=np.int32),  # 1D integer array
        np.array(["asdf", "qwer"]),  # string array
        np.array([True, False]),  # boolean array
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),  # 2D float array
        np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int16),  # 3D integer array
        np.array([np.nan, np.inf, -np.inf]),  # special float values
        {"arr": np.array([1, 2, 3]), "nested": {"arr": np.array([4, 5, 6])}},  # nested dict with arrays
        [np.array([1, 2]), np.array([3, 4])],  # list of arrays
        np.zeros((3, 4, 5), dtype=np.float32),  # 3D zeros
        np.ones((2, 3), dtype=np.float64),  # 2D ones with double precision
        {
            "__crfs__": {
                "noise": np.zeros((10, 32), dtype=np.float32),
                "intervention_step": 5,
                "intervention_mode": "latent_resume_edit",
                "resume_latent": np.arange(320, dtype=np.float32).reshape(10, 32),
                "resume_time": np.asarray(0.4999999, dtype=np.float32),
                "latent_edit": np.zeros((10, 32), dtype=np.float32),
                "latent_edit_space": "model",
                "return_trace": True,
                "return_normalized_final": True,
            }
        },  # exact R04B resume request, including scalar float32 time
        {
            "actions": np.zeros((10, 7), dtype=np.float32),
            "crfs_trace": {
                "step_index": np.asarray(5, dtype=np.int64),
                "time": np.asarray(0.4999999, dtype=np.float32),
                "x_t_pre_edit": np.zeros((10, 32), dtype=np.float32),
                "latent_edit": np.zeros((10, 32), dtype=np.float32),
                "x_t_post_edit": np.zeros((10, 32), dtype=np.float32),
                "v_post_edit": np.zeros((10, 32), dtype=np.float32),
                "predicted_clean_post_edit": np.zeros((10, 32), dtype=np.float32),
                "predicted_clean_post_edit_physical": np.zeros((10, 7), dtype=np.float32),
                "final_normalized": np.zeros((10, 32), dtype=np.float32),
            },
        },  # exact R04B response fields remain typed arrays over transport
    ],
)
def test_pack_unpack(data):
    packed = msgpack_numpy.packb(data)
    unpacked = msgpack_numpy.unpackb(packed)
    tree.map_structure(_check, data, unpacked)
