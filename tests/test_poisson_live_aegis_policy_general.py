from __future__ import annotations

from copy import deepcopy
import math
import unittest

from scripts import run_poisson_closed_loop_canary as runner


_IDENTITY = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]


def _shape(value):
    if not isinstance(value, list):
        return ()
    if not value:
        return (0,)
    child = _shape(value[0])
    if any(_shape(item) != child for item in value):
        raise ValueError("fake array requires a rectangular value")
    return (len(value),) + child


def _flatten(value):
    if isinstance(value, _Array):
        value = value.tolist()
    if isinstance(value, list):
        return [item for row in value for item in _flatten(row)]
    return [float(value)]


def _binary(left, right, operation):
    if isinstance(left, _Array):
        left = left.tolist()
    if isinstance(right, _Array):
        right = right.tolist()
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError("fake array shape mismatch")
        return [
            _binary(a, b, operation) for a, b in zip(left, right)
        ]
    return operation(float(left), float(right))


class _Array:
    def __init__(self, value):
        if isinstance(value, _Array):
            value = value.tolist()
        self._value = deepcopy(value)
        self.shape = _shape(self._value)

    def __iter__(self):
        return iter(self._value)

    def __getitem__(self, index):
        return self._value[index]

    def __sub__(self, other):
        return _Array(_binary(self, other, lambda left, right: left - right))

    def copy(self):
        return _Array(self._value)

    def tolist(self):
        return deepcopy(self._value)


class _FakeNumpy:
    float64 = float

    @staticmethod
    def asarray(value, dtype=None):
        del dtype
        return _Array(value)

    @staticmethod
    def isfinite(value):
        return all(math.isfinite(item) for item in _flatten(value))

    @staticmethod
    def all(value):
        return bool(value)

    @staticmethod
    def allclose(left, right, rtol=0.0, atol=1e-8):
        left_values = _flatten(left)
        right_values = _flatten(right)
        return len(left_values) == len(right_values) and all(
            abs(a - b) <= atol + rtol * abs(b)
            for a, b in zip(left_values, right_values)
        )

    class linalg:
        @staticmethod
        def norm(value):
            return math.sqrt(sum(item * item for item in _flatten(value)))


_NP = _FakeNumpy()


def _action(*, z_before, z_after, q1_diag):
    context = {
        "p1": [0.1, 0.2, 0.3],
        "R1": deepcopy(_IDENTITY),
        "q1_diag": list(q1_diag),
        "p2": [0.4, 0.5, 0.6],
        "R2": deepcopy(_IDENTITY),
        "Q2_diag": [0.03, 0.04, 0.05],
        "z_before": list(z_before),
        "status": "solved",
        "solver_status": "optimal",
        "z_after": list(z_after),
    }
    return {
        "executed": [0.0] * 7,
        "qp": {
            "status": "solved",
            "solver_status": "optimal",
            "z_before": list(z_before),
            "z_after": list(z_after),
            "context": context,
        },
    }


def _historical_result(action_count, *, first_q1_diag=(0.06, 0.12, 0.11)):
    actions = [
        _action(
            z_before=[1.0, 0.0, 0.0],
            z_after=[1.0, 0.0, 0.0],
            q1_diag=first_q1_diag,
        )
        for _ in range(action_count)
    ]
    return {
        "perception": {
            "status": "ready",
            "mvee_center": [0.4, 0.5, 0.6],
            "mvee_rotation": deepcopy(_IDENTITY),
            "mvee_semiaxes": [0.03, 0.04, 0.05],
        },
        "actions": actions,
    }


class _Client:
    def infer(self, _policy_input):
        return {"actions": [[0.0] * 7 for _ in range(10)]}


class _Evaluator:
    TABLE_POLICY_RESIZE = 224

    def __init__(self, first_context):
        self.first_context = deepcopy(first_context)
        self.observed_q1_diag = None
        self.observed_proxy = None

    @staticmethod
    def query_seed(seed, query_index):
        return int(seed) + int(query_index)

    @staticmethod
    def _policy_observation(
        _runtime, _observation, *, task_description, resize_size, rng_seed
    ):
        del task_description, resize_size, rng_seed
        return {"unused": True}

    @staticmethod
    def array_sha256(value):
        # Keep this dependency-free fixture on the same numeric float64
        # representation whether the host happens to have NumPy installed or
        # not.  Passing the private fake array directly to real NumPy creates
        # an object-dtype array whose pointer bytes are intentionally not the
        # released Table-1 numeric array hash.
        if isinstance(value, _Array):
            value = value.tolist()
        return runner._array_sha256(value)

    @staticmethod
    def translational_action(raw):
        return _NP.asarray(raw, dtype=_NP.float64).tolist()

    @staticmethod
    def _eef_proxy(_runtime, _observation):
        return {"p1": [0.0, 0.0, 0.0], "R1": deepcopy(_IDENTITY)}

    def _aegis_action(
        self,
        _runtime,
        *,
        nominal_translational,
        proxy,
        geometry,
        q1_diag,
        diagnostics_enabled,
    ):
        del geometry
        self.observed_proxy = {
            key: value.tolist() if hasattr(value, "tolist") else deepcopy(value)
            for key, value in proxy.items()
        }
        self.observed_q1_diag = q1_diag.copy()
        if diagnostics_enabled is not True:
            raise AssertionError("diagnostic AEGIS execution must remain enabled")
        return list(nominal_translational), {
            "status": "solved",
            "solver_status": "optimal",
            "context": deepcopy(self.first_context),
        }


def _provider(historical_result, *, source_start_action=None, first_query_index=0):
    requested_start = 180 if source_start_action is None else source_start_action
    context_index = (
        requested_start
        if requested_start < len(historical_result["actions"])
        else 0
    )
    kwargs = {
        "evaluator": _Evaluator(
            historical_result["actions"][context_index]["qp"]["context"]
        ),
        "runtime": {"np": _NP},
        "client": _Client(),
        "case": {"policy_noise_seed": 41},
        "task_description": "test task",
        "historical_result": historical_result,
        "first_query_index": first_query_index,
        "replan_steps": 5,
        "model_action_horizon": 10,
        "historical_first_chunk_sha256_diagnostic": "historical-chunk",
        "paired_first_query_cache": runner.PairedFirstQueryCache(
            first_query_index=first_query_index
        ),
        "first_query_execution": runner.FIRST_QUERY_LIVE_AND_CACHE,
        "paired_first_query_source_arm": None,
        "arm_name": "baseline",
    }
    if source_start_action is not None:
        kwargs["source_start_action"] = source_start_action
    return runner.LiveAegisPolicy(**kwargs)


class LiveAegisPolicyGeneralizationTests(unittest.TestCase):
    def test_action_zero_uses_its_historical_z_before_and_infers_count(self):
        historical = _historical_result(120)
        historical["actions"][0]["qp"]["z_before"] = [0.0, 1.0, 0.0]
        historical["actions"][0]["qp"]["context"]["z_before"] = [
            0.0,
            1.0,
            0.0,
        ]
        provider = _provider(historical, source_start_action=0)

        self.assertEqual(provider.initial_z_fixed, [0.0, 1.0, 0.0])
        self.assertEqual(provider.historical_action_count, 120)
        record = provider.record()
        self.assertEqual(record["source_start_action"], 0)
        self.assertEqual(record["historical_action_count"], 120)
        self.assertEqual(record["historical_first_action_index"], 0)
        self.assertEqual(
            record["initial_aegis_z_source"],
            {"historical_action_index": 0, "qp_field": "z_before"},
        )
        self.assertEqual(
            record["source"],
            "live_pi05_with_shared_current_q0_then_per_arm_own_observations",
        )
        self.assertNotIn(
            "initial_aegis_z_fixed_from_historical_action_179", record
        )

    def test_nonzero_start_uses_preceding_action_z_after(self):
        historical = _historical_result(4)
        historical["actions"][1]["qp"]["z_after"] = [0.0, 0.0, 1.0]
        provider = _provider(historical, source_start_action=2)

        self.assertEqual(provider.initial_z_fixed, [0.0, 0.0, 1.0])
        self.assertEqual(
            provider.record()["initial_aegis_z_source"],
            {"historical_action_index": 1, "qp_field": "z_after"},
        )

    def test_first_historical_context_supplies_milk_q1_diag(self):
        historical = _historical_result(
            3, first_q1_diag=(0.2, 0.2, 0.2)
        )
        provider = _provider(historical, source_start_action=0)
        provider._observation_record = lambda _env, _observation: {
            "policy_input_fingerprint_sha256": "settled-observation"
        }

        provider(
            env=object(),
            observation={},
            local_action_index=0,
            source_action_index=0,
        )

        self.assertEqual(
            provider.evaluator.observed_q1_diag.tolist(),
            [0.2, 0.2, 0.2],
        )
        self.assertEqual(
            provider.evaluator.observed_proxy["p1"],
            [0.1, 0.2, 0.3],
        )
        self.assertEqual(
            provider.record()["first_aegis_proxy_source"],
            "released_pre_settle_proxy_from_historical_action0_context",
        )
        self.assertGreater(
            provider.record()["first_fresh_proxy_diagnostic"][
                "p1_l2_difference_m"
            ],
            0.0,
        )
        provider(
            env=object(),
            observation={},
            local_action_index=1,
            source_action_index=1,
        )
        self.assertEqual(
            provider.evaluator.observed_proxy["p1"],
            [0.0, 0.0, 0.0],
        )
        self.assertEqual(
            [
                row["aegis_proxy_source"]
                for row in provider.record()["high_level_action_trace"]
            ],
            [
                "released_pre_settle_proxy_from_historical_action0_context",
                "current_observation_proxy",
            ],
        )
        self.assertEqual(
            provider.record()[
                "aegis_q1_diag_from_historical_first_action_context"
            ],
            [0.2, 0.2, 0.2],
        )

    def test_source_cadence_is_relative_to_generic_start(self):
        provider = _provider(
            _historical_result(3), source_start_action=0
        )
        with self.assertRaisesRegex(
            runner.ClosedLoopRunnerError,
            "expected 0, observed 180",
        ):
            provider(
                env=object(),
                observation={},
                local_action_index=0,
                source_action_index=180,
            )

    def test_default_action_180_retains_legacy_v1_record_fields(self):
        historical = _historical_result(181)
        historical["actions"][179]["qp"]["z_after"] = [0.0, 1.0, 0.0]
        provider = _provider(historical, first_query_index=36)
        record = provider.record()

        self.assertEqual(provider.source_start_action, 180)
        self.assertEqual(provider.historical_action_count, 181)
        self.assertEqual(
            record["source"],
            "live_pi05_with_shared_current_q36_then_per_arm_own_observations",
        )
        self.assertEqual(
            record["initial_aegis_z_fixed_from_historical_action_179"],
            [0.0, 1.0, 0.0],
        )
        self.assertEqual(
            record["historical_action_180_required_aegis_inputs"],
            record["historical_first_action_required_aegis_inputs"],
        )
        self.assertEqual(
            record["historical_action_180_aegis_full_output_diagnostic"],
            record["historical_first_action_aegis_full_output_diagnostic"],
        )

    def test_start_must_address_an_action_in_the_inferred_ledger(self):
        with self.assertRaisesRegex(
            runner.ClosedLoopRunnerError,
            "outside the historical AEGIS ledger",
        ):
            _provider(_historical_result(120))


if __name__ == "__main__":
    unittest.main()
