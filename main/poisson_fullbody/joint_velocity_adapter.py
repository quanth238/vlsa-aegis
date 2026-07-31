"""Matched translation-only VLA to seven-joint velocity adapter.

The released SafeLIBERO OSC controller clips normalized Cartesian commands to
``[-1, 1]`` and maps XYZ to a pose-goal delta of at most 0.05 metres.  This
module reproduces that target construction, then holds the target for one
50 ms high-level action while recomputing resolved-rate control at five 10 ms
updates.  The final gripper command is copied unchanged.

All velocity calculations use physical SI units.  Conversion to the
Robosuite ``JOINT_VELOCITY`` normalized action happens only after damped least
squares and physical joint-velocity clipping.
"""

from dataclasses import dataclass
import math


OSC_POSITION_DELTA_SCALE_M = 0.05
HIGH_LEVEL_DT_SECONDS = 0.05
INNER_DT_SECONDS = 0.01
INNER_UPDATES_PER_ACTION = 5
JV_PHYSICAL_LIMIT_RAD_S = 0.5


_NUMPY = None


def _numpy():
    global _NUMPY
    if _NUMPY is None:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - minimal baseline install
            raise RuntimeError(
                "the opt-in joint-velocity adapter requires NumPy"
            ) from exc
        _NUMPY = np
    return _NUMPY


def _readonly(array):
    np = _numpy()
    result = np.array(array, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _finite_vector(value, length, label):
    np = _numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,):
        raise ValueError(
            "{} must have shape ({},), got {}".format(
                label, length, array.shape
            )
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("{} must contain only finite values".format(label))
    return np.array(array, dtype=np.float64, copy=True)


def _rotation_matrix(value, label):
    np = _numpy()
    rotation = np.asarray(value, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("{} must have shape (3, 3)".format(label))
    if not np.all(np.isfinite(rotation)):
        raise ValueError("{} must contain only finite values".format(label))
    orthogonality_error = float(
        np.linalg.norm(rotation.T @ rotation - np.eye(3), ord=np.inf)
    )
    determinant = float(np.linalg.det(rotation))
    if orthogonality_error > 1.0e-8 or not math.isclose(
        determinant, 1.0, rel_tol=1.0e-8, abs_tol=1.0e-8
    ):
        raise ValueError("{} must be a proper rotation matrix".format(label))
    return np.array(rotation, dtype=np.float64, copy=True)


def _skew_vee(matrix):
    np = _numpy()
    return np.asarray(
        [matrix[2, 1], matrix[0, 2], matrix[1, 0]],
        dtype=np.float64,
    )


def rotation_vector_error(target_rotation, current_rotation):
    """Return world-frame rotation vector taking current into target.

    MuJoCo's rotational Jacobian rows describe world angular velocity.  The
    relative rotation is therefore ``target @ current.T`` and its logarithm
    is returned in world coordinates.  The near-zero and near-pi branches are
    handled without SciPy.
    """

    np = _numpy()
    target = _rotation_matrix(target_rotation, "target_rotation")
    current = _rotation_matrix(current_rotation, "current_rotation")
    relative = target @ current.T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)

    if angle < 1.0e-8:
        # vee((R-R.T)/2) = sin(theta) * axis and is theta*axis to
        # first order.
        return _skew_vee(0.5 * (relative - relative.T))
    if math.pi - angle < 1.0e-6:
        # At pi the skew part vanishes and the axis sign is intrinsically
        # ambiguous.  The principal eigenvector of (R+I)/2 is stable.
        symmetric = 0.5 * (relative + np.eye(3))
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        skew_hint = _skew_vee(relative - relative.T)
        if float(np.dot(axis, skew_hint)) < 0.0:
            axis = -axis
        return angle * axis
    return (
        angle
        / (2.0 * math.sin(angle))
        * _skew_vee(relative - relative.T)
    )


def _damped_least_squares_details(jacobian, desired_twist, damping):
    np = _numpy()
    matrix = np.asarray(jacobian, dtype=np.float64)
    if matrix.shape != (6, 7):
        raise ValueError("jacobian must have shape (6, 7)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("jacobian must contain only finite values")
    twist = _finite_vector(desired_twist, 6, "desired_twist")
    damping = float(damping)
    if not math.isfinite(damping) or damping <= 0.0:
        raise ValueError("damping must be finite and strictly positive")

    regularized = matrix @ matrix.T + damping * damping * np.eye(6)
    task_solution = np.linalg.solve(regularized, twist)
    qdot = matrix.T @ task_solution
    if not np.all(np.isfinite(qdot)):
        raise RuntimeError("damped least-squares solution is nonfinite")

    singular_values = np.linalg.svd(matrix, compute_uv=False)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    smallest = float(singular_values[-1]) if singular_values.size else 0.0
    threshold = max(matrix.shape) * np.finfo(np.float64).eps * largest
    rank = int(np.count_nonzero(singular_values > threshold))
    condition_number = None if smallest <= threshold else largest / smallest
    return qdot, singular_values, rank, condition_number


def damped_least_squares_velocity(jacobian, desired_twist, damping=0.05):
    """Return physical seven-joint velocity before actuator clipping."""

    qdot, _, _, _ = _damped_least_squares_details(
        jacobian, desired_twist, damping
    )
    return qdot


def normalized_joint_velocity_action(qdot_physical, gripper_command):
    """Encode a bounded physical command for Robosuite JOINT_VELOCITY.

    This function does not silently repair an out-of-range safety-filter
    result.  The caller must apply or verify physical bounds first.
    """

    np = _numpy()
    qdot = _finite_vector(qdot_physical, 7, "qdot_physical")
    gripper = float(gripper_command)
    if not math.isfinite(gripper):
        raise ValueError("gripper_command must be finite")
    tolerance = 64.0 * np.finfo(np.float64).eps
    if np.any(np.abs(qdot) > JV_PHYSICAL_LIMIT_RAD_S + tolerance):
        raise ValueError("qdot_physical exceeds the registered physical limit")
    normalized = np.clip(
        qdot / JV_PHYSICAL_LIMIT_RAD_S,
        -1.0,
        1.0,
    )
    return np.concatenate((normalized, [gripper]))


@dataclass(frozen=True)
class CartesianPoseTarget:
    """One fixed 50 ms target derived from a translation-only VLA action."""

    source_action: object
    clipped_translation: object
    ignored_rotation_components: object
    position_delta_m: object
    target_position: object
    target_rotation: object
    gripper_command: float

    def __post_init__(self):
        object.__setattr__(
            self,
            "source_action",
            _readonly(_finite_vector(self.source_action, 7, "source_action")),
        )
        object.__setattr__(
            self,
            "clipped_translation",
            _readonly(
                _finite_vector(
                    self.clipped_translation, 3, "clipped_translation"
                )
            ),
        )
        object.__setattr__(
            self,
            "ignored_rotation_components",
            _readonly(
                _finite_vector(
                    self.ignored_rotation_components,
                    3,
                    "ignored_rotation_components",
                )
            ),
        )
        object.__setattr__(
            self,
            "position_delta_m",
            _readonly(_finite_vector(self.position_delta_m, 3, "position_delta_m")),
        )
        object.__setattr__(
            self,
            "target_position",
            _readonly(_finite_vector(self.target_position, 3, "target_position")),
        )
        object.__setattr__(
            self,
            "target_rotation",
            _readonly(_rotation_matrix(self.target_rotation, "target_rotation")),
        )
        gripper = float(self.gripper_command)
        if not math.isfinite(gripper):
            raise ValueError("gripper_command must be finite")
        object.__setattr__(self, "gripper_command", gripper)

    def to_record(self):
        return {
            "source_action": self.source_action.tolist(),
            "clipped_translation": self.clipped_translation.tolist(),
            "ignored_rotation_components": (
                self.ignored_rotation_components.tolist()
            ),
            "position_delta_m": self.position_delta_m.tolist(),
            "target_position": self.target_position.tolist(),
            "target_rotation": self.target_rotation.tolist(),
            "gripper_command": self.gripper_command,
            "position_delta_scale_m": OSC_POSITION_DELTA_SCALE_M,
            "high_level_dt_seconds": HIGH_LEVEL_DT_SECONDS,
            "inner_dt_seconds": INNER_DT_SECONDS,
            "inner_updates": INNER_UPDATES_PER_ACTION,
            "translation_only": True,
        }


def create_translational_pose_target(
    vla_action,
    current_position,
    current_rotation,
):
    """Create the released-OSC-equivalent translation-only pose goal."""

    np = _numpy()
    action = _finite_vector(vla_action, 7, "vla_action")
    position = _finite_vector(current_position, 3, "current_position")
    rotation = _rotation_matrix(current_rotation, "current_rotation")
    clipped_translation = np.clip(action[:3], -1.0, 1.0)
    position_delta = OSC_POSITION_DELTA_SCALE_M * clipped_translation
    return CartesianPoseTarget(
        source_action=action,
        clipped_translation=clipped_translation,
        ignored_rotation_components=action[3:6],
        position_delta_m=position_delta,
        target_position=position + position_delta,
        # The registered translational comparison zeros the policy's rotation
        # components.  Its OSC orientation goal is therefore the pose at the
        # start of the 50 ms high-level command.
        target_rotation=rotation,
        gripper_command=action[6],
    )


@dataclass(frozen=True)
class AdapterStep:
    """One 10 ms nominal adapter command and JSON-ready diagnostics."""

    normalized_action: object
    qdot_physical: object
    qdot_unclipped: object
    desired_twist: object
    achieved_twist: object
    diagnostics: object

    def __post_init__(self):
        object.__setattr__(
            self,
            "normalized_action",
            _readonly(_finite_vector(self.normalized_action, 8, "normalized_action")),
        )
        object.__setattr__(
            self,
            "qdot_physical",
            _readonly(_finite_vector(self.qdot_physical, 7, "qdot_physical")),
        )
        object.__setattr__(
            self,
            "qdot_unclipped",
            _readonly(_finite_vector(self.qdot_unclipped, 7, "qdot_unclipped")),
        )
        object.__setattr__(
            self,
            "desired_twist",
            _readonly(_finite_vector(self.desired_twist, 6, "desired_twist")),
        )
        object.__setattr__(
            self,
            "achieved_twist",
            _readonly(_finite_vector(self.achieved_twist, 6, "achieved_twist")),
        )
        if not isinstance(self.diagnostics, dict):
            raise TypeError("diagnostics must be a dictionary")
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    def to_record(self):
        return {
            "normalized_action": self.normalized_action.tolist(),
            "qdot_physical_rad_s": self.qdot_physical.tolist(),
            "qdot_unclipped_rad_s": self.qdot_unclipped.tolist(),
            "desired_twist": self.desired_twist.tolist(),
            "achieved_twist": self.achieved_twist.tolist(),
            "diagnostics": dict(self.diagnostics),
        }


class TranslationalJointVelocityAdapter:
    """Stateful five-update adapter for one held Cartesian target at a time."""

    def __init__(
        self,
        damping=0.05,
        position_gain_s_inv=None,
        orientation_gain_s_inv=None,
    ):
        damping = float(damping)
        if not math.isfinite(damping) or damping <= 0.0:
            raise ValueError("damping must be finite and strictly positive")
        default_gain = 1.0 / HIGH_LEVEL_DT_SECONDS
        position_gain = (
            default_gain
            if position_gain_s_inv is None
            else float(position_gain_s_inv)
        )
        orientation_gain = (
            default_gain
            if orientation_gain_s_inv is None
            else float(orientation_gain_s_inv)
        )
        if not math.isfinite(position_gain) or position_gain <= 0.0:
            raise ValueError("position_gain_s_inv must be finite and positive")
        if not math.isfinite(orientation_gain) or orientation_gain <= 0.0:
            raise ValueError(
                "orientation_gain_s_inv must be finite and positive"
            )
        self.damping = damping
        self.position_gain_s_inv = position_gain
        self.orientation_gain_s_inv = orientation_gain
        self._target = None
        self._inner_update_index = 0
        self._previous_executed_qdot_physical = None
        self._pending_nominal_qdot_physical = None

    @property
    def target(self):
        return self._target

    @property
    def inner_update_index(self):
        return self._inner_update_index

    @property
    def updates_remaining(self):
        if self._target is None:
            return 0
        return max(0, INNER_UPDATES_PER_ACTION - self._inner_update_index)

    def reset(self):
        """Clear target and tracking state at an episode boundary."""

        self._target = None
        self._inner_update_index = 0
        self._previous_executed_qdot_physical = None
        self._pending_nominal_qdot_physical = None

    def begin_high_level_action(
        self,
        vla_action,
        current_position,
        current_rotation,
    ):
        """Freeze one Cartesian goal before the first of five inner updates."""

        if self._target is not None and self._inner_update_index < (
            INNER_UPDATES_PER_ACTION
        ):
            raise RuntimeError(
                "cannot replace a target before all five inner updates"
            )
        if self._pending_nominal_qdot_physical is not None:
            raise RuntimeError(
                "record the final executed joint velocity before a new target"
            )
        self._target = create_translational_pose_target(
            vla_action, current_position, current_rotation
        )
        self._inner_update_index = 0
        return self._target

    # Explicit alias for runner code that calls high-level actions targets.
    start_high_level_action = begin_high_level_action

    def compute_inner_action(
        self,
        current_position,
        current_rotation,
        eef_jacobian,
        measured_joint_velocity=None,
    ):
        """Recompute error, twist, and 8-D normalized action for one 10 ms step."""

        np = _numpy()
        if self._target is None:
            raise RuntimeError("begin_high_level_action must be called first")
        if self._inner_update_index >= INNER_UPDATES_PER_ACTION:
            raise RuntimeError(
                "the held target has already consumed all five inner updates"
            )
        if self._pending_nominal_qdot_physical is not None:
            raise RuntimeError(
                "record_executed_joint_velocity is required before the next update"
            )
        position = _finite_vector(current_position, 3, "current_position")
        rotation = _rotation_matrix(current_rotation, "current_rotation")
        jacobian = np.asarray(eef_jacobian, dtype=np.float64)
        if jacobian.shape != (6, 7) or not np.all(np.isfinite(jacobian)):
            raise ValueError("eef_jacobian must be a finite (6, 7) matrix")

        position_error = self._target.target_position - position
        orientation_error = rotation_vector_error(
            self._target.target_rotation, rotation
        )
        desired_twist = np.concatenate(
            (
                self.position_gain_s_inv * position_error,
                self.orientation_gain_s_inv * orientation_error,
            )
        )
        (
            qdot_unclipped,
            singular_values,
            jacobian_rank,
            condition_number,
        ) = _damped_least_squares_details(
            jacobian, desired_twist, self.damping
        )
        qdot_physical = np.clip(
            qdot_unclipped,
            -JV_PHYSICAL_LIMIT_RAD_S,
            JV_PHYSICAL_LIMIT_RAD_S,
        )
        normalized_action = normalized_joint_velocity_action(
            qdot_physical,
            self._target.gripper_command,
        )
        achieved_twist = jacobian @ qdot_physical
        twist_residual = desired_twist - achieved_twist
        saturated = np.abs(qdot_unclipped) > (
            JV_PHYSICAL_LIMIT_RAD_S
            + 64.0 * np.finfo(np.float64).eps
        )

        tracking_reference = self._previous_executed_qdot_physical
        if measured_joint_velocity is None:
            measured = None
            tracking_error = None
        else:
            measured = _finite_vector(
                measured_joint_velocity, 7, "measured_joint_velocity"
            )
            tracking_error = (
                None
                if tracking_reference is None
                else measured - tracking_reference
            )

        diagnostics = {
            "inner_update_index": self._inner_update_index,
            "inner_dt_seconds": INNER_DT_SECONDS,
            "high_level_dt_seconds": HIGH_LEVEL_DT_SECONDS,
            "updates_remaining_after_this": (
                INNER_UPDATES_PER_ACTION - self._inner_update_index - 1
            ),
            "target_position": self._target.target_position.tolist(),
            "position_error_m": position_error.tolist(),
            "position_error_norm_m": float(np.linalg.norm(position_error)),
            "orientation_error_rotvec_rad": orientation_error.tolist(),
            "orientation_error_norm_rad": float(
                np.linalg.norm(orientation_error)
            ),
            "position_gain_s_inv": self.position_gain_s_inv,
            "orientation_gain_s_inv": self.orientation_gain_s_inv,
            "damping": self.damping,
            "singular_values": singular_values.tolist(),
            "jacobian_rank": jacobian_rank,
            "jacobian_condition_number": condition_number,
            "saturated_joint_mask": saturated.tolist(),
            "saturated_joint_count": int(np.count_nonzero(saturated)),
            "twist_residual": twist_residual.tolist(),
            "twist_residual_norm": float(np.linalg.norm(twist_residual)),
            "joint_velocity_limit_rad_s": JV_PHYSICAL_LIMIT_RAD_S,
            "joint_velocity_normalization_scale_rad_s": (
                JV_PHYSICAL_LIMIT_RAD_S
            ),
            "gripper_command_unchanged": self._target.gripper_command,
            "tracking_reference_available": tracking_reference is not None,
            "measured_joint_velocity_rad_s": (
                None if measured is None else measured.tolist()
            ),
            "previous_commanded_joint_velocity_rad_s": (
                None
                if tracking_reference is None
                else tracking_reference.tolist()
            ),
            "previous_interval_tracking_error_rad_s": (
                None if tracking_error is None else tracking_error.tolist()
            ),
            "previous_interval_tracking_error_l2_rad_s": (
                None
                if tracking_error is None
                else float(np.linalg.norm(tracking_error))
            ),
            "previous_interval_tracking_error_linf_rad_s": (
                None
                if tracking_error is None
                else float(np.max(np.abs(tracking_error)))
            ),
        }

        self._pending_nominal_qdot_physical = np.array(
            qdot_physical, dtype=np.float64, copy=True
        )
        self._inner_update_index += 1
        return AdapterStep(
            normalized_action=normalized_action,
            qdot_physical=qdot_physical,
            qdot_unclipped=qdot_unclipped,
            desired_twist=desired_twist,
            achieved_twist=achieved_twist,
            diagnostics=diagnostics,
        )

    def record_executed_joint_velocity(self, qdot_physical):
        """Bind the actual post-filter command used for tracking diagnostics.

        The adapter computes a nominal velocity before the Poisson CBF-QP.
        Filtered arms may execute a different bounded velocity, so treating
        the nominal command as executed would corrupt the next update's
        tracking error.  Every computed update must be paired with this call,
        including the adapter-only arm where both vectors are identical.
        """

        np = _numpy()
        if self._pending_nominal_qdot_physical is None:
            raise RuntimeError("there is no pending nominal command to record")
        executed = _finite_vector(
            qdot_physical, 7, "executed_qdot_physical"
        )
        tolerance = 64.0 * np.finfo(np.float64).eps
        if np.any(np.abs(executed) > JV_PHYSICAL_LIMIT_RAD_S + tolerance):
            raise ValueError(
                "executed_qdot_physical exceeds the registered physical limit"
            )
        executed = np.clip(
            executed,
            -JV_PHYSICAL_LIMIT_RAD_S,
            JV_PHYSICAL_LIMIT_RAD_S,
        )
        nominal = self._pending_nominal_qdot_physical
        correction = executed - nominal
        self._previous_executed_qdot_physical = np.array(
            executed, dtype=np.float64, copy=True
        )
        self._pending_nominal_qdot_physical = None
        return {
            "nominal_qdot_physical_rad_s": nominal.tolist(),
            "executed_qdot_physical_rad_s": executed.tolist(),
            "filter_correction_rad_s": correction.tolist(),
            "filter_correction_l2_rad_s": float(np.linalg.norm(correction)),
            "normalized_executed_action": normalized_joint_velocity_action(
                executed, self._target.gripper_command
            ).tolist(),
        }

    # Short runner spelling; cadence checks remain identical.
    step = compute_inner_action


# The shorter name is convenient in experiment configuration without hiding
# that this is specifically the registered translation-only comparison.
JointVelocityAdapter = TranslationalJointVelocityAdapter
