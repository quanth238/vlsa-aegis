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

    def test_predictive_flow_guidance_is_validated_and_removed(self):
        nominal = [[0.0] * 7 for _ in range(10)]
        row = [0.0] * 30
        row[0] = 1.0
        guidance = {
            "schema_version": "crfs_predictive_flow_guidance.v1",
            "action_horizon": 10,
            "action_dimensions": [0, 1, 2],
            "nominal_output_actions": nominal,
            "delta_rows": [row],
            "delta_lower": [-0.5],
            "projection_sweeps": 64,
            "projection_tolerance": 5.0e-5,
        }
        observation = {
            "state": [1, 2, 3],
            "__crfs__": {"rng_seed": 17, "flow_guidance": guidance},
        }

        returned, control = self.module._extract_crfs_control(observation)

        self.assertNotIn("__crfs__", returned)
        self.assertEqual(control["rng_seed"], 17)
        self.assertEqual(control["flow_guidance"], guidance)

    def test_predictive_flow_guidance_rejects_zero_and_nonfinite_rows(self):
        base = {
            "schema_version": "crfs_predictive_flow_guidance.v1",
            "action_horizon": 10,
            "action_dimensions": [0, 1, 2],
            "nominal_output_actions": [[0.0] * 7 for _ in range(10)],
            "delta_rows": [[0.0] * 30],
            "delta_lower": [0.0],
            "projection_sweeps": 64,
            "projection_tolerance": 5.0e-5,
        }
        with self.assertRaises(ValueError):
            self.module._extract_crfs_control(
                {"__crfs__": {"rng_seed": 1, "flow_guidance": base}}
            )
        base["delta_rows"][0][2] = float("nan")
        with self.assertRaises(ValueError):
            self.module._extract_crfs_control(
                {"__crfs__": {"rng_seed": 1, "flow_guidance": base}}
            )

    def test_fixed_repulsive_flow_guidance_is_validated_and_removed(self):
        guidance = {
            "schema_version": "crfs_fixed_repulsive_flow_guidance.v1",
            "action_horizon": 10,
            "action_dimensions": [0, 1, 2],
            "physical_output_direction": [1.0, 0.0, 0.0],
            "nominal_output_actions": [[0.0] * 7 for _ in range(10)],
            "guided_action_slots": [2, 3, 4],
            "guided_euler_steps": [5, 6, 7, 8, 9],
            "step_size_action": 0.05,
            "action_limit": 1.0,
        }
        returned, control = self.module._extract_crfs_control(
            {
                "state": [1, 2, 3],
                "__crfs__": {
                    "rng_seed": 19,
                    "repulsive_flow_guidance": guidance,
                },
            }
        )
        self.assertNotIn("__crfs__", returned)
        self.assertEqual(control["repulsive_flow_guidance"], guidance)

    def test_fixed_repulsive_flow_guidance_rejects_wrong_slots(self):
        guidance = {
            "schema_version": "crfs_fixed_repulsive_flow_guidance.v1",
            "action_horizon": 10,
            "action_dimensions": [0, 1, 2],
            "physical_output_direction": [1.0, 0.0, 0.0],
            "nominal_output_actions": [[0.0] * 7 for _ in range(10)],
            "guided_action_slots": [1, 2, 3],
            "guided_euler_steps": [5, 6, 7, 8, 9],
            "step_size_action": 0.05,
            "action_limit": 1.0,
        }
        with self.assertRaises(ValueError):
            self.module._extract_crfs_control(
                {
                    "__crfs__": {
                        "rng_seed": 19,
                        "repulsive_flow_guidance": guidance,
                    }
                }
            )

    def test_scheduled_repulsive_flow_guidance_is_validated(self):
        guidance = {
            "schema_version": "crfs_scheduled_repulsive_flow_guidance.v1",
            "action_horizon": 10,
            "action_dimensions": [0, 1, 2],
            "physical_output_direction": [1.0, 0.0, 0.0],
            "nominal_output_actions": [[0.0] * 7 for _ in range(10)],
            "guided_action_slots": [2, 3, 4],
            "euler_step_strengths_action": [0.0] * 8 + [1.0 / 12.0, 1.0 / 6.0],
            "action_limit": 1.0,
        }
        returned, control = self.module._extract_crfs_control(
            {
                "state": [1, 2, 3],
                "__crfs__": {
                    "rng_seed": 19,
                    "scheduled_repulsive_flow_guidance": guidance,
                },
            }
        )
        self.assertNotIn("__crfs__", returned)
        self.assertEqual(
            control["scheduled_repulsive_flow_guidance"], guidance
        )
    def test_embodisteer_task_metric_guidance_is_validated_and_removed(self):
        row = [0.0] * 30
        row[0] = 1.0
        guidance = {
            "schema_version": "crfs_embodisteer_multicbf_guidance.v1",
            "action_horizon": 10,
            "action_dimensions": [0, 1, 2],
            "nominal_output_actions": [[0.0] * 7 for _ in range(10)],
            "delta_rows": [row],
            "delta_lower": [-0.5],
            "task_metric_directions": [row],
            "task_metric_condition_number": 2.0,
            "guidance_schedule": {
                "base_strength": 1.0,
                "beta": 50.0,
                "transition": 0.7,
            },
            "projection_sweeps": 64,
            "projection_tolerance": 5.0e-5,
        }
        observation = {
            "state": [1, 2, 3],
            "__crfs__": {
                "rng_seed": 23,
                "embodisteer_guidance": guidance,
            },
        }

        returned, control = self.module._extract_crfs_control(observation)

        self.assertNotIn("__crfs__", returned)
        self.assertEqual(control["rng_seed"], 23)
        self.assertEqual(control["embodisteer_guidance"], guidance)

    def test_guidance_payloads_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            self.module._extract_crfs_control(
                {
                    "__crfs__": {
                        "rng_seed": 1,
                        "flow_guidance": {},
                        "embodisteer_guidance": {},
                    }
                }
            )

    def test_embodisteer_joint_denoising_primitives_are_validated(self):
        initialize = {
            "schema_version": "crfs_embodisteer_joint_denoising.v1",
            "mode": "initialize",
            "action_horizon": 10,
            "num_steps": 10,
        }
        returned, control = self.module._extract_crfs_control(
            {
                "state": [1, 2, 3],
                "__crfs__": {
                    "rng_seed": 31,
                    "embodisteer_joint_denoising": initialize,
                },
            }
        )
        self.assertNotIn("__crfs__", returned)
        self.assertEqual(control["embodisteer_joint_denoising"], initialize)

        step = {
            "schema_version": "crfs_embodisteer_joint_denoising.v1",
            "mode": "step",
            "action_horizon": 10,
            "num_steps": 10,
            "time": 0.7,
            "model_actions": [[0.0] * 32 for _ in range(10)],
            "physical_pose_actions": [[0.0] * 6 for _ in range(10)],
        }
        _, control = self.module._extract_crfs_control(
            {
                "__crfs__": {
                    "rng_seed": 31,
                    "embodisteer_joint_denoising": step,
                }
            }
        )
        self.assertEqual(control["embodisteer_joint_denoising"], step)

    def test_joint_denoising_rejects_bad_shapes_and_other_payloads(self):
        step = {
            "schema_version": "crfs_embodisteer_joint_denoising.v1",
            "mode": "step",
            "action_horizon": 10,
            "num_steps": 10,
            "time": 1.0,
            "model_actions": [[0.0] * 31 for _ in range(10)],
            "physical_pose_actions": [[0.0] * 6 for _ in range(10)],
        }
        with self.assertRaises(ValueError):
            self.module._extract_crfs_control(
                {
                    "__crfs__": {
                        "rng_seed": 1,
                        "embodisteer_joint_denoising": step,
                    }
                }
            )
        with self.assertRaises(ValueError):
            self.module._extract_crfs_control(
                {
                    "__crfs__": {
                        "rng_seed": 1,
                        "flow_guidance": {},
                        "embodisteer_joint_denoising": {},
                    }
                }
            )
