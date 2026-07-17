import importlib.util
from pathlib import Path
import sys
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = ROOT / "openpi/src/openpi/serving/websocket_policy_server.py"


def _load_server_module():
    """Load the server helper without requiring the full OpenPI environment."""

    openpi_client = types.ModuleType("openpi_client")
    openpi_client.base_policy = types.SimpleNamespace(BasePolicy=object)
    openpi_client.msgpack_numpy = types.SimpleNamespace()
    sys.modules.setdefault("openpi_client", openpi_client)
    sys.modules.setdefault("openpi_client.base_policy", openpi_client.base_policy)
    sys.modules.setdefault("openpi_client.msgpack_numpy", openpi_client.msgpack_numpy)

    websockets = types.ModuleType("websockets")
    websockets.ConnectionClosed = type("ConnectionClosed", (Exception,), {})
    websockets_asyncio = types.ModuleType("websockets.asyncio")
    websockets_server = types.ModuleType("websockets.asyncio.server")
    websockets_server.ServerConnection = object
    websockets_server.Request = object
    websockets_server.Response = object
    websockets_frames = types.ModuleType("websockets.frames")
    websockets_frames.CloseCode = types.SimpleNamespace(
        INTERNAL_ERROR=1011
    )
    websockets.asyncio = websockets_asyncio
    websockets.frames = websockets_frames
    websockets_asyncio.server = websockets_server
    sys.modules.setdefault("websockets", websockets)
    sys.modules.setdefault("websockets.asyncio", websockets_asyncio)
    sys.modules.setdefault("websockets.asyncio.server", websockets_server)
    sys.modules.setdefault("websockets.frames", websockets_frames)

    spec = importlib.util.spec_from_file_location("crfs_websocket_server", SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CrfsWebsocketControlTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_server_module()

    def test_ordinary_observation_is_unchanged(self):
        observation = {"prompt": "move the bowl", "state": [1, 2, 3]}

        returned, control = self.module._extract_crfs_control(
            observation.copy()
        )

        self.assertEqual(returned, observation)
        self.assertIsNone(control)

    def test_reserved_control_is_removed_before_policy_transforms(self):
        observation = {
            "prompt": "move the bowl",
            "state": [1, 2, 3],
            "__crfs__": {"rng_seed": 42},
        }

        returned, control = self.module._extract_crfs_control(observation)

        self.assertNotIn("__crfs__", returned)
        self.assertEqual(control, {"rng_seed": 42})

    def test_invalid_controls_fail_closed(self):
        invalid_controls = [
            ("42", TypeError),
            ({"rng_seed": True}, TypeError),
            ({"rng_seed": -1}, ValueError),
            ({"rng_seed": 2**32}, ValueError),
            ({"rng_seed": 1, "noise": []}, ValueError),
            ({}, ValueError),
        ]
        for control, error in invalid_controls:
            with self.subTest(control=control):
                with self.assertRaises(error):
                    self.module._extract_crfs_control(
                        {"__crfs__": control}
                    )
