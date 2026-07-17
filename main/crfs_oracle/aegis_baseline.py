"""Fail-closed adapter for the byte-preserved upstream VLSA/AEGIS layer.

This module is additive.  It does not import or modify ``main/main_aegis.py``
or ``main/utils.py``.  The two runtime dependencies used by the safety filter,
NumPy and CVXPY, are imported only when the numeric adapter is constructed or
executed so dependency-free contract tests can import this module.

The literal upstream runner captures the end-effector proxy pose *before* its
20 dummy settling actions and does not refresh it before the first safety QP.
``pre_settle_upstream`` intentionally preserves that behavior.  Initializing
from the current branch pose is exposed only as the separately named
``current_state_corrected_diagnostic`` and must not be reported as literal
upstream AEGIS.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence


UPSTREAM_BASELINE_COMMIT = "57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b"
UPSTREAM_SOURCE_SHA256 = MappingProxyType(
    {
        "main/main_aegis.py": "8491b9381dd0d6dbd1ba9bd0360db7c93a932d7e5913e517b4f2291abd64042c",
        "main/main_aegis_translational.py": "2d397b3969ec50fc1811fe6037ecbfc3594202d5511d736a89314e8e9b41f0a7",
        "main/utils.py": "d2efc608eb17f47e246cac8f0a8cbef364889e89d2b4ad0f302b1d55b45b2ecf",
    }
)

PERCEPTION_ORIGINAL_END_TO_END = "original_end_to_end"
PERCEPTION_FROZEN_LABEL_CORE = "frozen_label_aegis_core"
INITIALIZATION_PRE_SETTLE_UPSTREAM = "pre_settle_upstream"
INITIALIZATION_CURRENT_STATE_DIAGNOSTIC = "current_state_corrected_diagnostic"
ZHIPUAI_API_KEY_ENV = "ZHIPUAI_API_KEY"
GROUNDING_DINO_CONFIG_ENV = "AEGIS_GROUNDING_DINO_CONFIG"
GROUNDING_DINO_CHECKPOINT_ENV = "AEGIS_GROUNDING_DINO_CHECKPOINT"

UPSTREAM_DT = 0.05
UPSTREAM_ALPHA_GAIN = 10.0
UPSTREAM_ACTION_TO_VELOCITY = 5.0
UPSTREAM_VELOCITY_TO_ACTION = 0.2
UPSTREAM_EEF_LOCAL_OFFSET = (0.0, 0.0, -0.08)
UPSTREAM_EEF_Q_DIAG_STANDARD = (0.06, 0.12, 0.11)
UPSTREAM_EEF_Q_DIAG_TALL = (0.06, 0.12, 0.2)
UPSTREAM_OBSTACLE_L1_THRESHOLD_M = 0.001

LOW_MOTION_EEF_PATH_RATIO_MAX = 0.10
LOW_MOTION_ABS_PROGRESS_MAX_M = 0.005
STOP_LIKE_EEF_PATH_MAX_M = 0.005
STOP_LIKE_ABS_PROGRESS_MAX_M = 0.001


class AegisBaselineError(RuntimeError):
    """Typed fail-closed error suitable for a terminal case artifact."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, object] | None = None,
    ) -> None:
        self.code = str(code)
        self.message = str(message)
        self.details = dict(details or {})
        super().__init__(f"{self.code}: {self.message}")

    def as_failure(self) -> dict[str, object]:
        return {
            "status": "failed",
            "failure_code": self.code,
            "message": self.message,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class PerceptionContract:
    mode: str
    semantic_label_source: str
    obstacle_label: str | None
    dino_config_path: str | None
    dino_checkpoint_path: str | None
    zhipu_key_present: bool
    runtime_dependencies_checked: bool


@dataclass(frozen=True)
class AegisFilterResult:
    action: tuple[float, ...]
    telemetry: Mapping[str, object]


@dataclass(frozen=True)
class ActionModificationMetrics:
    action_steps: int
    modified_steps: int
    modification_tolerance: float
    total_delta_l2: float
    mean_step_delta_l2: float
    max_step_delta_l2: float
    translation_delta_l2: float
    rotation_delta_l2: float
    gripper_delta_l2: float
    nominal_translation_command_path: float
    aegis_translation_command_path: float
    translation_command_path_ratio: float | None
    nominal_full_action_l2: float
    aegis_full_action_l2: float
    full_action_norm_ratio: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StoppingDiagnostic:
    baseline_eef_path_m: float
    aegis_eef_path_m: float
    eef_path_ratio: float | None
    eef_path_reduction_fraction: float | None
    baseline_progress_m: float
    aegis_progress_m: float
    progress_retention_ratio: float | None
    absolute_aegis_progress_m: float
    low_motion_eef_path_ratio_max: float
    low_motion_absolute_progress_max_m: float
    ratio_low_motion_diagnostic: bool
    stop_like_eef_path_max_m: float
    stop_like_absolute_progress_max_m: float
    stop_like: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class UpstreamObstacleDisplacementDiagnostic:
    l1_displacement_m: float
    threshold_m: float
    upstream_collision_proxy: bool
    physical_contact_measurement: bool
    simulator_clearance_measurement: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _QpProblem:
    a_u_v: Any
    a_u_omega: Any
    a_u_z: Any
    h: float
    u_reference: Any
    weight: Any


@dataclass(frozen=True)
class _QpSolution:
    values: Any | None
    status: str
    objective_value: float | None


def _repo_root(repo_root: str | Path | None = None) -> Path:
    if repo_root is not None:
        return Path(repo_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def verify_upstream_sources(repo_root: str | Path | None = None) -> dict[str, str]:
    """Verify every upstream AEGIS source used to define this adapter."""

    root = _repo_root(repo_root)
    actual: dict[str, str] = {}
    for relative_path, expected_digest in UPSTREAM_SOURCE_SHA256.items():
        path = root / relative_path
        if not path.is_file():
            raise AegisBaselineError(
                "upstream_source_drift",
                "required upstream AEGIS source is missing",
                details={"path": relative_path, "expected_sha256": expected_digest},
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        actual[relative_path] = digest
        if digest != expected_digest:
            raise AegisBaselineError(
                "upstream_source_drift",
                "upstream AEGIS source hash does not match the frozen baseline",
                details={
                    "path": relative_path,
                    "expected_sha256": expected_digest,
                    "actual_sha256": digest,
                    "upstream_commit": UPSTREAM_BASELINE_COMMIT,
                },
            )
    return actual


def _lazy_import(module_name: str, *, failure_code: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as exc:
        raise AegisBaselineError(
            failure_code,
            f"required runtime dependency {module_name!r} is unavailable",
            details={"module": module_name, "exception_type": type(exc).__name__},
        ) from exc


def _load_numpy() -> Any:
    return _lazy_import("numpy", failure_code="runtime_dependency_failure")


def _check_original_perception_dependencies() -> None:
    _lazy_import("zai", failure_code="perception_failure")
    _lazy_import("groundingdino.util.inference", failure_code="perception_failure")


def resolve_perception_mode(
    mode: str,
    *,
    obstacle_label: str | None = None,
    environ: Mapping[str, str] | None = None,
    require_runtime_dependencies: bool = False,
) -> PerceptionContract:
    """Resolve the semantic-perception contract without silently degrading.

    ``frozen_label_aegis_core`` consumes already validated obstacle geometry and
    a frozen semantic label.  It is intentionally distinct from the original
    Zhipu + GroundingDINO end-to-end pipeline.
    """

    if mode == PERCEPTION_FROZEN_LABEL_CORE:
        label = "" if obstacle_label is None else str(obstacle_label).strip()
        if not label:
            raise AegisBaselineError(
                "perception_failure",
                "frozen-label AEGIS-core mode requires a non-empty obstacle label",
                details={"mode": mode},
            )
        return PerceptionContract(
            mode=mode,
            semantic_label_source="frozen_manifest_or_case_label",
            obstacle_label=label,
            dino_config_path=None,
            dino_checkpoint_path=None,
            zhipu_key_present=False,
            runtime_dependencies_checked=False,
        )

    if mode != PERCEPTION_ORIGINAL_END_TO_END:
        raise AegisBaselineError(
            "perception_failure",
            "unknown AEGIS perception mode",
            details={"mode": mode},
        )

    if obstacle_label is not None:
        raise AegisBaselineError(
            "perception_failure",
            "original end-to-end mode must obtain its label from Zhipu, not a frozen label",
            details={"mode": mode},
        )

    environment = os.environ if environ is None else environ
    if not str(environment.get(ZHIPUAI_API_KEY_ENV, "")).strip():
        raise AegisBaselineError(
            "perception_failure",
            f"original end-to-end AEGIS requires {ZHIPUAI_API_KEY_ENV}",
            details={"mode": mode, "missing": [ZHIPUAI_API_KEY_ENV]},
        )

    config_value = str(environment.get(GROUNDING_DINO_CONFIG_ENV, "")).strip()
    checkpoint_value = str(environment.get(GROUNDING_DINO_CHECKPOINT_ENV, "")).strip()
    missing_variables = [
        name
        for name, value in (
            (GROUNDING_DINO_CONFIG_ENV, config_value),
            (GROUNDING_DINO_CHECKPOINT_ENV, checkpoint_value),
        )
        if not value
    ]
    if missing_variables:
        raise AegisBaselineError(
            "perception_failure",
            "original end-to-end AEGIS requires explicit GroundingDINO asset paths",
            details={"mode": mode, "missing": missing_variables},
        )
    dino_config = Path(config_value).expanduser().resolve()
    dino_checkpoint = Path(checkpoint_value).expanduser().resolve()
    for label, path, expected_name in (
        ("config", dino_config, "GroundingDINO_SwinT_OGC.py"),
        ("checkpoint", dino_checkpoint, "groundingdino_swint_ogc.pth"),
    ):
        if not path.is_file():
            raise AegisBaselineError(
                "perception_failure",
                f"GroundingDINO {label} path is not a file",
                details={"mode": mode, "path": str(path)},
            )
        if path.name != expected_name:
            raise AegisBaselineError(
                "perception_failure",
                f"GroundingDINO {label} has the wrong basename",
                details={
                    "mode": mode,
                    "path": str(path),
                    "expected_basename": expected_name,
                },
            )
    if require_runtime_dependencies:
        _check_original_perception_dependencies()

    return PerceptionContract(
        mode=mode,
        semantic_label_source="zhipu_glm_4_5v",
        obstacle_label=None,
        dino_config_path=str(dino_config),
        dino_checkpoint_path=str(dino_checkpoint),
        zhipu_key_present=True,
        runtime_dependencies_checked=bool(require_runtime_dependencies),
    )


def upstream_eef_q_diag(task_description: str) -> tuple[float, float, float]:
    """Return the exact task-string branch from upstream ``main_aegis.py``."""

    description = str(task_description)
    if any(name in description for name in ("orange juice", "milk", "alphabet soup")):
        return UPSTREAM_EEF_Q_DIAG_TALL
    return UPSTREAM_EEF_Q_DIAG_STANDARD


def _vector_hat(np: Any, value: Any) -> Any:
    return np.array(
        [
            [0.0, -value[2], value[1]],
            [value[2], 0.0, -value[0]],
            [-value[1], value[0], 0.0],
        ]
    )


def _project_matrix(np: Any, z: Any) -> Any:
    normalized = z / (np.linalg.norm(z) + 1e-12)
    return np.eye(3) - np.outer(normalized, normalized)


def _compute_h_ij(
    np: Any,
    p_i: Any,
    q_i_diag: Any,
    r_i: Any,
    p_j: Any,
    q_j_diag: Any,
    r_j: Any,
    z_ij: Any,
) -> float:
    q_i = np.diag(q_i_diag)
    q_j = np.diag(q_j_diag)
    qbar_i = r_i @ q_i @ r_i.T
    qbar_j = r_j @ q_j @ r_j.T
    qbar_i_inv = np.linalg.inv(qbar_i)
    z = z_ij / np.linalg.norm(z_ij)
    term1 = np.linalg.norm(qbar_j @ qbar_i_inv @ z)
    term2 = (p_j - p_i).T @ qbar_i_inv @ z
    denom = np.linalg.norm(qbar_i_inv @ z)
    return float((-term1 + term2 - 1.0) / denom)


def _compute_h_coeffs_3d(
    np: Any,
    p_i: Any,
    q_i_diag: Any,
    r_i: Any,
    p_j: Any,
    q_j_diag: Any,
    r_j: Any,
    z: Any,
    eps: float = 1e-10,
) -> tuple[Any, Any, Any, float, Any]:
    """Literal arithmetic from upstream ``utils.compute_h_coeffs_3d``."""

    q_i = np.diag(q_i_diag)
    q_j = np.diag(q_j_diag)
    qbar_i = r_i @ q_i @ r_i.T
    qbar_j = r_j @ q_j @ r_j.T
    qbar_i_inv = np.linalg.inv(qbar_i)
    qbar_i_inv2 = qbar_i_inv @ qbar_i_inv
    qbar_j2 = qbar_j @ qbar_j

    z = z / (np.linalg.norm(z) + eps)
    a_vec = qbar_i_inv @ z
    denom = np.linalg.norm(a_vec) + eps
    b_vec = qbar_j @ a_vec
    term1 = np.linalg.norm(b_vec) + eps
    sigma = term1 * denom + eps
    rho = 1.0 - (p_j - p_i).T @ a_vec + term1

    eta_row = -(1.0 / denom) * (z.T @ qbar_i_inv)
    term_mu_1 = (rho / (denom**3 + eps)) * (z.T @ qbar_i_inv2)
    term_mu_2 = (1.0 / denom) * ((p_j - p_i).T @ qbar_i_inv)
    term_mu_3 = (1.0 / sigma) * (z.T @ qbar_i_inv @ qbar_j2 @ qbar_i_inv)
    mu_row = term_mu_1 + term_mu_2 - term_mu_3

    tmp1 = z.T @ qbar_i_inv2 @ _vector_hat(np, z)
    left_vec = z.T @ qbar_i_inv @ qbar_j2
    ja_vec = _vector_hat(np, a_vec)
    tmp2 = left_vec @ (ja_vec - qbar_i_inv @ _vector_hat(np, z))
    part_a = (p_j - p_i).T @ qbar_i_inv @ _vector_hat(np, z)
    part_b = z.T @ qbar_i_inv @ _vector_hat(np, p_j - p_i)
    tmp3 = part_a + part_b
    zeta_tilde = (
        rho * (1.0 / (denom**3 + eps)) * tmp1
        + (1.0 / sigma) * tmp2
        + (1.0 / denom) * tmp3
    )

    a_v = (eta_row @ r_i).ravel()
    a_omega = zeta_tilde @ r_i
    a_u_z = (mu_row @ _project_matrix(np, z)).ravel()
    h = _compute_h_ij(np, p_i, q_i_diag, r_i, p_j, q_j_diag, r_j, z)
    return a_v, a_omega, a_u_z, h, mu_row


def _solve_upstream_qp(np: Any, problem: _QpProblem) -> _QpSolution:
    """Build the exact full nine-variable CVXPY/OSQP problem."""

    cp = _lazy_import("cvxpy", failure_code="runtime_dependency_failure")
    u = cp.Variable(9)
    objective = cp.Minimize(cp.quad_form(u - problem.u_reference, problem.weight))
    constraints = [
        problem.a_u_v @ u[:3]
        + problem.a_u_omega @ u[3:6]
        + problem.a_u_z @ u[6:]
        + UPSTREAM_ALPHA_GAIN * problem.h
        >= 0
    ]
    qp = cp.Problem(objective, constraints)
    try:
        qp.solve(solver=cp.OSQP)
    except Exception as exc:
        raise AegisBaselineError(
            "qp_failure",
            "upstream AEGIS OSQP solve raised an exception",
            details={"exception_type": type(exc).__name__},
        ) from exc
    objective_value = None if qp.value is None else float(qp.value)
    return _QpSolution(
        values=None if u.value is None else np.asarray(u.value, dtype=np.float64),
        status=str(qp.status),
        objective_value=objective_value,
    )


def _as_vector(np: Any, value: object, size: int, name: str) -> Any:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not bool(np.all(np.isfinite(array))):
        raise AegisBaselineError(
            "input_failure",
            f"{name} must be a finite vector of shape ({size},)",
            details={"name": name, "shape": list(array.shape)},
        )
    return array.copy()


def _as_rotation(np: Any, value: object, name: str) -> Any:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3, 3) or not bool(np.all(np.isfinite(array))):
        raise AegisBaselineError(
            "input_failure",
            f"{name} must be a finite 3-by-3 matrix",
            details={"name": name, "shape": list(array.shape)},
        )
    return array.copy()


def _json_value(value: Any) -> object:
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value) if isinstance(value, float) else value
    return value


class AegisSafetyCore:
    """Stateful adapter for the full upstream AEGIS action filter."""

    def __init__(
        self,
        *,
        pre_settle_p1: Sequence[float],
        pre_settle_r1: Sequence[Sequence[float]],
        branch_p1: Sequence[float],
        branch_r1: Sequence[Sequence[float]],
        q1_diag: Sequence[float],
        p2: Sequence[float],
        q2_diag: Sequence[float],
        r2: Sequence[Sequence[float]],
        perception: PerceptionContract,
        initialization_source: str = INITIALIZATION_PRE_SETTLE_UPSTREAM,
        repo_root: str | Path | None = None,
        _coefficient_backend: (
            Callable[..., tuple[Any, Any, Any, float, Any]] | None
        ) = None,
        _qp_backend: Callable[[Any, _QpProblem], _QpSolution] | None = None,
    ) -> None:
        self._source_hashes = verify_upstream_sources(repo_root)
        self._np = _load_numpy()
        if not isinstance(perception, PerceptionContract):
            raise AegisBaselineError(
                "perception_failure",
                "AegisSafetyCore requires a validated PerceptionContract",
            )
        if initialization_source not in {
            INITIALIZATION_PRE_SETTLE_UPSTREAM,
            INITIALIZATION_CURRENT_STATE_DIAGNOSTIC,
        }:
            raise AegisBaselineError(
                "initialization_failure",
                "unknown AEGIS initialization source",
                details={"initialization_source": initialization_source},
            )

        np = self._np
        self._pre_settle_p1 = _as_vector(np, pre_settle_p1, 3, "pre_settle_p1")
        self._pre_settle_r1 = _as_rotation(np, pre_settle_r1, "pre_settle_r1")
        self._branch_p1 = _as_vector(np, branch_p1, 3, "branch_p1")
        self._branch_r1 = _as_rotation(np, branch_r1, "branch_r1")
        self._q1_diag = _as_vector(np, q1_diag, 3, "q1_diag")
        self._p2 = _as_vector(np, p2, 3, "p2")
        self._q2_diag = _as_vector(np, q2_diag, 3, "q2_diag")
        self._r2 = _as_rotation(np, r2, "r2")
        if bool(np.any(self._q1_diag <= 0.0)) or bool(np.any(self._q2_diag <= 0.0)):
            raise AegisBaselineError(
                "input_failure",
                "AEGIS ellipsoid diagonal entries must be positive",
            )

        if initialization_source == INITIALIZATION_PRE_SETTLE_UPSTREAM:
            self._p1 = self._pre_settle_p1.copy()
            self._r1 = self._pre_settle_r1.copy()
        else:
            self._p1 = self._branch_p1.copy()
            self._r1 = self._branch_r1.copy()

        z = self._p2 - self._p1
        z_norm = float(np.linalg.norm(z))
        if not math.isfinite(z_norm) or z_norm <= 0.0:
            raise AegisBaselineError(
                "geometry_failure",
                "upstream AEGIS cannot initialize z from coincident proxy centers",
            )
        self._z = z / z_norm
        self._perception = perception
        self._initialization_source = initialization_source
        self._coefficient_backend = _coefficient_backend
        self._qp_backend = _qp_backend
        self._step_index = 0

        relative_rotation = self._pre_settle_r1.T @ self._branch_r1
        cosine = float((np.trace(relative_rotation) - 1.0) / 2.0)
        cosine = max(-1.0, min(1.0, cosine))
        self._initialization_telemetry = {
            "initialization_source": initialization_source,
            "literal_upstream_initialization": initialization_source
            == INITIALIZATION_PRE_SETTLE_UPSTREAM,
            "pre_settle_p1": _json_value(self._pre_settle_p1),
            "branch_p1": _json_value(self._branch_p1),
            "pre_settle_to_branch_translation_delta_m": float(
                np.linalg.norm(self._branch_p1 - self._pre_settle_p1)
            ),
            "pre_settle_to_branch_rotation_frobenius": float(
                np.linalg.norm(self._branch_r1 - self._pre_settle_r1)
            ),
            "pre_settle_to_branch_rotation_angle_rad": float(math.acos(cosine)),
            "initial_p1_used": _json_value(self._p1),
            "initial_r1_used": _json_value(self._r1),
            "initial_z": _json_value(self._z),
        }

    @property
    def initialization_telemetry(self) -> dict[str, object]:
        return dict(self._initialization_telemetry)

    @property
    def z(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self._z)

    def refresh_after_action(
        self,
        *,
        eef_position: Sequence[float],
        eef_rotation_matrix: Sequence[Sequence[float]],
    ) -> dict[str, object]:
        """Apply the post-action p1/R1 refresh performed by upstream AEGIS."""

        np = self._np
        eef_position_array = _as_vector(np, eef_position, 3, "eef_position")
        rotation = _as_rotation(np, eef_rotation_matrix, "eef_rotation_matrix")
        offset = np.asarray(UPSTREAM_EEF_LOCAL_OFFSET, dtype=np.float64)
        proxy_center = eef_position_array + rotation @ offset
        self._p1 = proxy_center
        self._r1 = rotation
        return {
            "refresh_source": "post_action_upstream_eef_pose",
            "p1": _json_value(self._p1),
            "r1": _json_value(self._r1),
        }

    def filter_action(self, nominal_action: Sequence[float]) -> AegisFilterResult:
        """Filter one seven-dimensional LIBERO action or fail explicitly."""

        np = self._np
        action = _as_vector(np, nominal_action, 7, "nominal_action")
        p1_before = self._p1.copy()
        r1_before = self._r1.copy()
        z_before = self._z.copy()

        coefficient_backend = self._coefficient_backend or _compute_h_coeffs_3d
        try:
            a_v, a_omega, a_u_z, h, mu_row = coefficient_backend(
                np,
                p1_before,
                self._q1_diag,
                r1_before,
                self._p2,
                self._q2_diag,
                self._r2,
                z_before,
            )
            a_v = _as_vector(np, a_v, 3, "a_v")
            a_omega = _as_vector(np, a_omega, 3, "a_omega")
            a_u_z = _as_vector(np, a_u_z, 3, "a_u_z")
            mu_row = _as_vector(np, mu_row, 3, "mu_row")
            h_value = float(h)
            if not math.isfinite(h_value):
                raise ValueError("non-finite h")
        except AegisBaselineError:
            raise
        except Exception as exc:
            raise AegisBaselineError(
                "geometry_failure",
                "AEGIS CBF coefficient construction failed",
                details={"exception_type": type(exc).__name__},
            ) from exc

        v_reference = r1_before.T @ action[:3]
        u_v_reference = UPSTREAM_ACTION_TO_VELOCITY * v_reference
        omega_reference = action[3:6]
        u_omega_reference = UPSTREAM_ACTION_TO_VELOCITY * omega_reference
        a_u_v = UPSTREAM_VELOCITY_TO_ACTION * a_v
        a_u_omega = UPSTREAM_VELOCITY_TO_ACTION * a_omega
        u_z_nominal = UPSTREAM_ALPHA_GAIN * mu_row
        weight = np.diag([1.0 / 25.0] * 6 + [1.0, 1.0, 1.0])
        u_reference = np.hstack([u_v_reference, u_omega_reference, u_z_nominal])
        problem = _QpProblem(
            a_u_v=a_u_v,
            a_u_omega=a_u_omega,
            a_u_z=a_u_z,
            h=h_value,
            u_reference=u_reference,
            weight=weight,
        )

        qp_backend = self._qp_backend or _solve_upstream_qp
        try:
            solution = qp_backend(np, problem)
        except AegisBaselineError:
            raise
        except Exception as exc:
            raise AegisBaselineError(
                "qp_failure",
                "AEGIS QP backend raised an exception",
                details={"exception_type": type(exc).__name__},
            ) from exc
        if not isinstance(solution, _QpSolution):
            raise AegisBaselineError(
                "qp_failure",
                "AEGIS QP backend returned an invalid result type",
                details={"result_type": type(solution).__name__},
            )
        status = str(solution.status).lower()
        if solution.values is None or status not in {"optimal", "optimal_inaccurate"}:
            raise AegisBaselineError(
                "qp_failure",
                "AEGIS QP produced no accepted solution; nominal fallback is forbidden",
                details={"solver_status": str(solution.status)},
            )
        values = _as_vector(np, solution.values, 9, "qp_solution")
        objective_value = solution.objective_value
        if objective_value is not None and not math.isfinite(float(objective_value)):
            raise AegisBaselineError(
                "qp_failure",
                "AEGIS QP objective is non-finite",
                details={"solver_status": str(solution.status)},
            )

        u_v = values[:3]
        u_omega = values[3:6]
        u_z = values[6:]
        identity = np.eye(len(z_before))
        dz = (identity - np.outer(z_before, z_before)) @ u_z
        z_after = z_before + dz * UPSTREAM_DT
        z_after_norm = float(np.linalg.norm(z_after))
        if not math.isfinite(z_after_norm) or z_after_norm <= 0.0:
            raise AegisBaselineError(
                "geometry_failure",
                "AEGIS z update produced a zero or non-finite direction",
            )
        z_after = z_after / z_after_norm

        filtered = np.zeros(7)
        filtered[:3] = UPSTREAM_VELOCITY_TO_ACTION * r1_before @ u_v
        filtered[3:6] = UPSTREAM_VELOCITY_TO_ACTION * u_omega
        filtered[6] = action[6]
        if not bool(np.all(np.isfinite(filtered))):
            raise AegisBaselineError(
                "qp_failure",
                "AEGIS QP produced a non-finite filtered action",
            )

        constraint_lhs = float(
            a_u_v @ u_v
            + a_u_omega @ u_omega
            + a_u_z @ u_z
            + UPSTREAM_ALPHA_GAIN * h_value
        )
        self._z = z_after
        telemetry = {
            "step_index": self._step_index,
            "perception_mode": self._perception.mode,
            "semantic_label_source": self._perception.semantic_label_source,
            "obstacle_label": self._perception.obstacle_label,
            "initialization": self.initialization_telemetry,
            "upstream_source_sha256": dict(self._source_hashes),
            "qp_backend": (
                "cvxpy_osqp_upstream"
                if self._qp_backend is None
                else "injected_test_backend"
            ),
            "coefficient_backend": (
                "upstream_literal"
                if self._coefficient_backend is None
                else "injected_test_backend"
            ),
            "solver_status": str(solution.status),
            "objective_value": (
                None if objective_value is None else float(objective_value)
            ),
            "p1_before": _json_value(p1_before),
            "r1_before": _json_value(r1_before),
            "z_before": _json_value(z_before),
            "z_after": _json_value(z_after),
            "dz": _json_value(dz),
            "h": h_value,
            "a_v": _json_value(a_v),
            "a_omega": _json_value(a_omega),
            "a_u_v": _json_value(a_u_v),
            "a_u_omega": _json_value(a_u_omega),
            "a_u_z": _json_value(a_u_z),
            "mu_row": _json_value(mu_row),
            "u_z_nominal": _json_value(u_z_nominal),
            "u_reference": _json_value(u_reference),
            "u_solution": _json_value(values),
            "constraint_lhs": constraint_lhs,
            "nominal_action": _json_value(action),
            "filtered_action": _json_value(filtered),
            "action_delta": _json_value(filtered - action),
            "gripper_preserved_exactly": bool(filtered[6] == action[6]),
            "upstream_obstacle_collision_proxy": "external_l1_displacement_helper",
            "scientific_collision_measurement": "external_contact_and_d_sim_required",
        }
        self._step_index += 1
        return AegisFilterResult(
            action=tuple(float(value) for value in filtered),
            telemetry=telemetry,
        )


def _finite_rows(
    actions: Sequence[Sequence[float]], name: str
) -> tuple[tuple[float, ...], ...]:
    rows: list[tuple[float, ...]] = []
    for index, row in enumerate(actions):
        values = tuple(float(value) for value in row)
        if len(values) != 7 or not all(math.isfinite(value) for value in values):
            raise AegisBaselineError(
                "metric_input_failure",
                f"{name}[{index}] must contain exactly seven finite values",
            )
        rows.append(values)
    if not rows:
        raise AegisBaselineError(
            "metric_input_failure", f"{name} must contain at least one action"
        )
    return tuple(rows)


def _l2(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _optional_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


def action_modification_metrics(
    nominal_actions: Sequence[Sequence[float]],
    aegis_actions: Sequence[Sequence[float]],
    *,
    modification_tolerance: float = 0.0,
) -> ActionModificationMetrics:
    """Compute continuous action-change diagnostics without NumPy."""

    tolerance = float(modification_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise AegisBaselineError(
            "metric_input_failure",
            "modification_tolerance must be finite and non-negative",
        )
    nominal = _finite_rows(nominal_actions, "nominal_actions")
    aegis = _finite_rows(aegis_actions, "aegis_actions")
    if len(nominal) != len(aegis):
        raise AegisBaselineError(
            "metric_input_failure",
            "nominal and AEGIS action sequences must have equal length",
            details={"nominal_steps": len(nominal), "aegis_steps": len(aegis)},
        )

    all_deltas: list[float] = []
    translation_deltas: list[float] = []
    rotation_deltas: list[float] = []
    gripper_deltas: list[float] = []
    step_delta_norms: list[float] = []
    nominal_flat: list[float] = []
    aegis_flat: list[float] = []
    nominal_translation_path = 0.0
    aegis_translation_path = 0.0
    modified_steps = 0
    for nominal_row, aegis_row in zip(nominal, aegis):
        delta = [
            aegis_value - nominal_value
            for nominal_value, aegis_value in zip(nominal_row, aegis_row)
        ]
        delta_norm = _l2(delta)
        step_delta_norms.append(delta_norm)
        modified_steps += int(delta_norm > tolerance)
        all_deltas.extend(delta)
        translation_deltas.extend(delta[:3])
        rotation_deltas.extend(delta[3:6])
        gripper_deltas.append(delta[6])
        nominal_flat.extend(nominal_row)
        aegis_flat.extend(aegis_row)
        nominal_translation_path += _l2(nominal_row[:3])
        aegis_translation_path += _l2(aegis_row[:3])

    return ActionModificationMetrics(
        action_steps=len(nominal),
        modified_steps=modified_steps,
        modification_tolerance=tolerance,
        total_delta_l2=_l2(all_deltas),
        mean_step_delta_l2=sum(step_delta_norms) / len(step_delta_norms),
        max_step_delta_l2=max(step_delta_norms),
        translation_delta_l2=_l2(translation_deltas),
        rotation_delta_l2=_l2(rotation_deltas),
        gripper_delta_l2=_l2(gripper_deltas),
        nominal_translation_command_path=nominal_translation_path,
        aegis_translation_command_path=aegis_translation_path,
        translation_command_path_ratio=_optional_ratio(
            aegis_translation_path, nominal_translation_path
        ),
        nominal_full_action_l2=_l2(nominal_flat),
        aegis_full_action_l2=_l2(aegis_flat),
        full_action_norm_ratio=_optional_ratio(_l2(aegis_flat), _l2(nominal_flat)),
    )


def stopping_diagnostic(
    *,
    baseline_eef_path_m: float,
    aegis_eef_path_m: float,
    baseline_progress_m: float,
    aegis_progress_m: float,
) -> StoppingDiagnostic:
    """Apply the frozen stopping diagnostic; this is not a safety metric."""

    values = {
        "baseline_eef_path_m": float(baseline_eef_path_m),
        "aegis_eef_path_m": float(aegis_eef_path_m),
        "baseline_progress_m": float(baseline_progress_m),
        "aegis_progress_m": float(aegis_progress_m),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise AegisBaselineError(
            "metric_input_failure", "stopping diagnostic inputs must be finite"
        )
    if values["baseline_eef_path_m"] < 0.0 or values["aegis_eef_path_m"] < 0.0:
        raise AegisBaselineError(
            "metric_input_failure", "EEF path lengths must be non-negative"
        )
    path_ratio = _optional_ratio(
        values["aegis_eef_path_m"], values["baseline_eef_path_m"]
    )
    progress_ratio = _optional_ratio(
        values["aegis_progress_m"], values["baseline_progress_m"]
    )
    ratio_low_motion = bool(
        path_ratio is not None
        and path_ratio <= LOW_MOTION_EEF_PATH_RATIO_MAX
        and abs(values["aegis_progress_m"]) <= LOW_MOTION_ABS_PROGRESS_MAX_M
    )
    stop_like = bool(
        values["aegis_eef_path_m"] <= STOP_LIKE_EEF_PATH_MAX_M
        and abs(values["aegis_progress_m"]) <= STOP_LIKE_ABS_PROGRESS_MAX_M
    )
    return StoppingDiagnostic(
        baseline_eef_path_m=values["baseline_eef_path_m"],
        aegis_eef_path_m=values["aegis_eef_path_m"],
        eef_path_ratio=path_ratio,
        eef_path_reduction_fraction=None if path_ratio is None else 1.0 - path_ratio,
        baseline_progress_m=values["baseline_progress_m"],
        aegis_progress_m=values["aegis_progress_m"],
        progress_retention_ratio=progress_ratio,
        absolute_aegis_progress_m=abs(values["aegis_progress_m"]),
        low_motion_eef_path_ratio_max=LOW_MOTION_EEF_PATH_RATIO_MAX,
        low_motion_absolute_progress_max_m=LOW_MOTION_ABS_PROGRESS_MAX_M,
        ratio_low_motion_diagnostic=ratio_low_motion,
        stop_like_eef_path_max_m=STOP_LIKE_EEF_PATH_MAX_M,
        stop_like_absolute_progress_max_m=STOP_LIKE_ABS_PROGRESS_MAX_M,
        stop_like=stop_like,
    )


def upstream_obstacle_displacement_diagnostic(
    initial_position: Sequence[float], current_position: Sequence[float]
) -> UpstreamObstacleDisplacementDiagnostic:
    """Reproduce upstream's L1 > 1 mm proxy without calling it contact."""

    initial = tuple(float(value) for value in initial_position)
    current = tuple(float(value) for value in current_position)
    if len(initial) != 3 or len(current) != 3:
        raise AegisBaselineError(
            "metric_input_failure", "obstacle positions must be three-dimensional"
        )
    if not all(math.isfinite(value) for value in initial + current):
        raise AegisBaselineError(
            "metric_input_failure", "obstacle positions must be finite"
        )
    displacement = sum(abs(after - before) for before, after in zip(initial, current))
    return UpstreamObstacleDisplacementDiagnostic(
        l1_displacement_m=displacement,
        threshold_m=UPSTREAM_OBSTACLE_L1_THRESHOLD_M,
        upstream_collision_proxy=displacement > UPSTREAM_OBSTACLE_L1_THRESHOLD_M,
        physical_contact_measurement=False,
        simulator_clearance_measurement=False,
    )


__all__ = [
    "AegisBaselineError",
    "AegisFilterResult",
    "AegisSafetyCore",
    "ActionModificationMetrics",
    "INITIALIZATION_CURRENT_STATE_DIAGNOSTIC",
    "INITIALIZATION_PRE_SETTLE_UPSTREAM",
    "GROUNDING_DINO_CHECKPOINT_ENV",
    "GROUNDING_DINO_CONFIG_ENV",
    "LOW_MOTION_ABS_PROGRESS_MAX_M",
    "LOW_MOTION_EEF_PATH_RATIO_MAX",
    "PERCEPTION_FROZEN_LABEL_CORE",
    "PERCEPTION_ORIGINAL_END_TO_END",
    "PerceptionContract",
    "STOP_LIKE_ABS_PROGRESS_MAX_M",
    "STOP_LIKE_EEF_PATH_MAX_M",
    "StoppingDiagnostic",
    "UPSTREAM_BASELINE_COMMIT",
    "UPSTREAM_OBSTACLE_L1_THRESHOLD_M",
    "UPSTREAM_SOURCE_SHA256",
    "UpstreamObstacleDisplacementDiagnostic",
    "action_modification_metrics",
    "resolve_perception_mode",
    "stopping_diagnostic",
    "upstream_eef_q_diag",
    "upstream_obstacle_displacement_diagnostic",
    "verify_upstream_sources",
    "ZHIPUAI_API_KEY_ENV",
]
