"""Strict, dependency-free R05A full-lifetime host telemetry parser.

The producer is a shell monitor running inside the source H100 allocation.  It
is intentionally not trusted to summarize its own trace: this module reparses
the sealed TSV with Python integer arithmetic and returns the only normalized
summary that a result-envelope finalizer or CPU ``afterany`` validator should
consume.

This parser validates an *accepted* trace.  A partial or diagnostic failure
trace is useful forensic evidence, but it cannot be converted into an accepted
summary by this API.
"""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import re
from typing import Any


TRACE_SCHEMA_VERSION = "1.0"
TRACE_ARTIFACT_ROLE = "r05a_full_lifetime_cgroup_v2_current_trace"
SUMMARY_SCHEMA_VERSION = "1.0"
SUMMARY_ARTIFACT_ROLE = "r05a_full_lifetime_cgroup_v2_current_summary"
REQUESTED_INTERVAL_MS = 100
MAX_ADJACENT_GAP_NS = 500_000_000
MAX_TRACE_BYTES = 32 * 1024 * 1024
MAX_TRACE_LINES = 100_000
MAX_LINE_BYTES = 4096

RECORD_PREFIX_KEYS = (
    "schema_version",
    "artifact_role",
    "slurm_array_job_id",
    "slurm_array_task_id",
    "proc_cgroup_line",
    "membership_path",
    "selected_mountinfo_line",
    "mount_root",
    "mount_point",
    "memory_events_hierarchy_state",
    "membership_relative_to_mount_root",
    "job_boundary_path",
    "kernel_osrelease",
)
LIFECYCLE_MARKERS = (
    "monitor_ready",
    "policy_launch",
    "policy_cleanup_complete",
    "gpu_monitor_cleanup_complete",
    "workload_cleanup_complete",
    "monitor_stop_observed",
)
CRITICAL_EVENT_KEYS = ("max", "oom", "oom_kill")
REQUIRED_EVENT_KEYS = frozenset(("low", "high", *CRITICAL_EVENT_KEYS))
SUMMARY_KEYS = (
    "job_scope_path",
    "job_memory_max_before_bytes",
    "job_memory_max_after_bytes",
    "sample_count_observed",
    "sample_interval_requested_seconds",
    "first_sample_monotonic_ns",
    "last_sample_monotonic_ns",
    "sample_max_gap_ns",
    "sampled_memory_current_high_water_bytes",
    "memory_events_max_delta",
    "memory_events_oom_delta",
    "memory_events_oom_kill_delta",
    "native_memory_peak_state",
    "native_memory_peak_value",
    "first_sample_no_later_than_policy_launch",
    "last_sample_no_earlier_than_workload_cleanup",
    "host_cgroup_sampled_current_is_lower_bound_not_peak",
    "outcome",
)

_CANONICAL_UINT = re.compile(r"0|[1-9][0-9]*")
_EVENT_KEY = re.compile(r"[a-z][a-z0-9_]*")
_SCALAR_STATES = frozenset(
    {
        "readable_integer",
        "readable_unlimited",
        "missing",
        "unreadable",
        "read_error",
        "readable_invalid",
    }
)
_EVENT_STATES = frozenset(
    {"readable_flat_keys", "missing", "unreadable", "readable_invalid"}
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _uint(text: str, *, label: str, positive: bool = False) -> int:
    if _CANONICAL_UINT.fullmatch(text) is None:
        raise ValueError(f"{label} is not a canonical unsigned integer")
    value = int(text)
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _decode(value: str) -> str:
    decoded: list[str] = []
    index = 0
    escapes = {"\\": "\\", "t": "\t", "n": "\n"}
    while index < len(value):
        if value[index] != "\\":
            decoded.append(value[index])
            index += 1
            continue
        if index + 1 >= len(value) or value[index + 1] not in escapes:
            raise ValueError("trace contains a noncanonical escape")
        decoded.append(escapes[value[index + 1]])
        index += 2
    return "".join(decoded)


def _decode_mount_path(value: str) -> str:
    """Decode the four escapes allowed in proc mountinfo path fields."""

    replacements = {"040": " ", "011": "\t", "012": "\n", "134": "\\"}
    decoded: list[str] = []
    index = 0
    while index < len(value):
        if value[index] != "\\":
            decoded.append(value[index])
            index += 1
            continue
        code = value[index + 1 : index + 4]
        if len(code) != 3 or code not in replacements:
            raise ValueError("selected mountinfo path contains an invalid escape")
        decoded.append(replacements[code])
        index += 4
    return "".join(decoded)


def _absolute_path(value: str, *, label: str) -> str:
    if not value.startswith("/"):
        raise ValueError(f"{label} is not absolute")
    if value != "/" and value.endswith("/"):
        raise ValueError(f"{label} has a trailing slash")
    if "//" in value:
        raise ValueError(f"{label} contains an empty component")
    if value == "/":
        return value
    components = value.split("/")[1:]
    if any(component in {"", ".", ".."} for component in components):
        raise ValueError(f"{label} is not normalized")
    if str(PurePosixPath(value)) != value:
        raise ValueError(f"{label} is not normalized")
    return value


def _suffix(path: str, root: str) -> str:
    if root == "/":
        return "" if path == "/" else path
    if path == root:
        return ""
    if path.startswith(root + "/"):
        return path[len(root) :]
    raise ValueError("cgroup membership is outside the selected mount root")


def _mapped_path(mount_point: str, suffix: str) -> str:
    if not suffix:
        return mount_point
    if mount_point == "/":
        return suffix
    return mount_point + suffix


def _parse_mountinfo_line(line: str) -> tuple[str, str, str]:
    fields = line.split(" ")
    if "" in fields:
        raise ValueError("selected mountinfo line has ambiguous whitespace")
    try:
        separator = fields.index("-")
    except ValueError as error:
        raise ValueError("selected mountinfo line has no separator") from error
    if separator < 6 or separator + 3 >= len(fields):
        raise ValueError("selected mountinfo line is truncated")
    if fields[separator + 1] != "cgroup2":
        raise ValueError("selected mountinfo line is not cgroup v2")
    mount_root = _absolute_path(
        _decode_mount_path(fields[3]), label="selected mount root"
    )
    mount_point = _absolute_path(
        _decode_mount_path(fields[4]), label="selected mount point"
    )
    option_fields = (fields[5], fields[separator + 3])
    options = {
        option
        for option_field in option_fields
        for option in option_field.split(",")
    }
    if "memory_localevents" in options:
        raise ValueError("selected cgroup2 mount has local-only memory.events")
    return mount_root, mount_point, "hierarchical"


def _record_once(
    destination: dict[Any, Any], key: Any, value: Any, *, label: str
) -> None:
    if key in destination:
        raise ValueError(f"trace duplicates {label}")
    destination[key] = value


def parse_full_lifetime_telemetry(
    path: str | Path, *, expected_job_id: str | int
) -> dict[str, Any]:
    """Parse and independently validate one sealed full-lifetime TSV.

    ``expected_job_id`` is the externally bound Slurm array job id.  The
    function raises :class:`ValueError` for every unsupported, ambiguous, or
    internally inconsistent artifact and otherwise returns a JSON-compatible
    normalized summary.
    """

    expected_job_text = str(expected_job_id)
    _uint(expected_job_text, label="expected job id", positive=True)
    trace_path = Path(path)
    try:
        raw = trace_path.read_bytes()
    except OSError as error:
        raise ValueError(f"telemetry trace cannot be read: {error}") from error
    if not raw or len(raw) > MAX_TRACE_BYTES:
        raise ValueError("telemetry trace is empty or exceeds its byte bound")
    if b"\x00" in raw or b"\r" in raw or not raw.endswith(b"\n"):
        raise ValueError("telemetry trace is not canonical newline-delimited UTF-8")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("telemetry trace is not UTF-8") from error
    lines = text.splitlines()
    if not lines or len(lines) > MAX_TRACE_LINES:
        raise ValueError("telemetry trace exceeds its line bound")
    if any(len(line.encode("utf-8")) > MAX_LINE_BYTES for line in lines):
        raise ValueError("telemetry trace contains an oversized line")

    records: dict[str, str] = {}
    record_order: list[str] = []
    scopes: dict[int, tuple[str, str, str, str]] = {}
    scope_row_order: list[int] = []
    metrics: dict[tuple[int, str], tuple[str, str]] = {}
    counters: dict[tuple[int, str, str], int] = {}
    snapshots: dict[str, tuple[str, str]] = {}
    events: dict[str, dict[str, int]] = {"before": {}, "after": {}}
    local_events: dict[str, dict[str, int]] = {"before": {}, "after": {}}
    lifecycles: dict[int, tuple[str, int]] = {}
    lifecycle_row_order: list[int] = []
    samples: dict[int, tuple[int, int]] = {}
    sample_row_order: list[int] = []
    capabilities: dict[str, tuple[str, str]] = {}
    summaries: dict[str, str] = {}
    summary_row_order: list[str] = []
    positions: dict[tuple[str, str], int] = {}
    event_positions: dict[str, list[int]] = {"before": [], "after": []}
    local_event_positions: dict[str, list[int]] = {"before": [], "after": []}

    for line_index, line in enumerate(lines):
        fields = line.split("\t")
        row_type = fields[0] if fields else ""
        if row_type == "record" and len(fields) == 3:
            key = fields[1]
            _record_once(records, key, _decode(fields[2]), label=f"record {key}")
            record_order.append(key)
            positions[("record", key)] = line_index
        elif row_type == "scope" and len(fields) == 6:
            index = _uint(fields[1], label="scope index")
            _record_once(
                scopes,
                index,
                (fields[2], _decode(fields[3]), _decode(fields[4]), fields[5]),
                label=f"scope {index}",
            )
            scope_row_order.append(index)
        elif row_type == "metric" and len(fields) == 5:
            index = _uint(fields[1], label="metric scope index")
            key = (index, fields[2])
            _record_once(
                metrics, key, (fields[3], _decode(fields[4])), label=f"metric {key}"
            )
        elif row_type == "counter" and len(fields) == 5:
            index = _uint(fields[1], label="counter scope index")
            if _EVENT_KEY.fullmatch(fields[3]) is None:
                raise ValueError("counter key is invalid")
            key = (index, fields[2], fields[3])
            _record_once(
                counters,
                key,
                _uint(fields[4], label=f"counter {key}"),
                label=f"counter {key}",
            )
        elif row_type == "snapshot" and len(fields) == 5:
            phase, metric = fields[1], fields[2]
            if phase not in {"before", "after"} or metric != "memory.max":
                raise ValueError("snapshot identity changed")
            _record_once(
                snapshots,
                phase,
                (fields[3], _decode(fields[4])),
                label=f"snapshot {phase}",
            )
            positions[("snapshot", phase)] = line_index
        elif row_type == "event" and len(fields) == 4:
            phase, key = fields[1], fields[2]
            if phase not in events or _EVENT_KEY.fullmatch(key) is None:
                raise ValueError("event row identity is invalid")
            _record_once(
                events[phase],
                key,
                _uint(fields[3], label=f"event {phase}.{key}"),
                label=f"event {phase}.{key}",
            )
            event_positions[phase].append(line_index)
        elif row_type == "local_event" and len(fields) == 4:
            phase, key = fields[1], fields[2]
            if phase not in local_events or _EVENT_KEY.fullmatch(key) is None:
                raise ValueError("local-event row identity is invalid")
            _record_once(
                local_events[phase],
                key,
                _uint(fields[3], label=f"local event {phase}.{key}"),
                label=f"local event {phase}.{key}",
            )
            local_event_positions[phase].append(line_index)
        elif row_type == "lifecycle" and len(fields) == 4:
            index = _uint(fields[1], label="lifecycle index")
            _record_once(
                lifecycles,
                index,
                (
                    fields[2],
                    _uint(
                        fields[3],
                        label=f"lifecycle {fields[2]} timestamp",
                        positive=True,
                    ),
                ),
                label=f"lifecycle {index}",
            )
            lifecycle_row_order.append(index)
            positions[("lifecycle", str(index))] = line_index
        elif row_type == "sample" and len(fields) == 4:
            index = _uint(fields[1], label="sample index")
            _record_once(
                samples,
                index,
                (
                    _uint(fields[2], label=f"sample {index} timestamp", positive=True),
                    _uint(fields[3], label=f"sample {index} bytes", positive=True),
                ),
                label=f"sample {index}",
            )
            sample_row_order.append(index)
            positions[("sample", str(index))] = line_index
        elif row_type == "capability" and len(fields) == 4:
            key = fields[1]
            _record_once(
                capabilities,
                key,
                (fields[2], _decode(fields[3])),
                label=f"capability {key}",
            )
            positions[("capability", key)] = line_index
        elif row_type == "summary" and len(fields) == 3:
            key = fields[1]
            _record_once(summaries, key, _decode(fields[2]), label=f"summary {key}")
            summary_row_order.append(key)
            positions[("summary", key)] = line_index
        else:
            raise ValueError(f"telemetry trace row {line_index + 1} has an unknown shape")

    if record_order != [*RECORD_PREFIX_KEYS, "status"]:
        raise ValueError("trace record keys or order changed")
    if positions.get(("record", "status")) != len(lines) - 1:
        raise ValueError("telemetry trace has no terminal status row")
    expected_records = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "artifact_role": TRACE_ARTIFACT_ROLE,
        "slurm_array_job_id": expected_job_text,
        "slurm_array_task_id": "0",
        "memory_events_hierarchy_state": "hierarchical",
        "status": "completed",
    }
    for key, expected in expected_records.items():
        if records.get(key) != expected:
            raise ValueError(f"trace record {key} changed")

    membership = _absolute_path(records["membership_path"], label="membership path")
    if records["proc_cgroup_line"] != f"0::{membership}":
        raise ValueError("proc cgroup line and membership path differ")
    selected_root, selected_point, hierarchy = _parse_mountinfo_line(
        records["selected_mountinfo_line"]
    )
    mount_root = _absolute_path(records["mount_root"], label="mount root")
    mount_point = _absolute_path(records["mount_point"], label="mount point")
    if (mount_root, mount_point, records["memory_events_hierarchy_state"]) != (
        selected_root,
        selected_point,
        hierarchy,
    ):
        raise ValueError("selected mountinfo mapping differs from trace records")
    membership_suffix = _suffix(membership, mount_root)
    if records["membership_relative_to_mount_root"] != membership_suffix:
        raise ValueError("membership-relative mount suffix is inconsistent")

    job_component = f"job_{expected_job_text}"
    components = membership.split("/")[1:]
    if components.count(job_component) != 1:
        raise ValueError("membership does not contain one exact job boundary")
    job_position = components.index(job_component)
    job_boundary = "/" + "/".join(components[: job_position + 1])
    if records["job_boundary_path"] != job_boundary:
        raise ValueError("recorded job boundary differs from exact source job")
    expected_logical: list[str] = []
    logical = membership
    while True:
        expected_logical.append(logical)
        if logical == job_boundary:
            break
        logical = logical.rsplit("/", 1)[0]
        if not logical or len(logical) < len(job_boundary):
            raise ValueError("scope chain escaped the exact job boundary")
    if sorted(scopes) != list(range(len(expected_logical))) or scope_row_order != list(
        range(len(expected_logical))
    ):
        raise ValueError("scope indices are not contiguous")
    for index, logical in enumerate(expected_logical):
        kind = (
            "task"
            if index == 0
            else "job"
            if logical == job_boundary
            else "step"
            if logical.rsplit("/", 1)[-1] == "step_batch"
            else "intermediate"
        )
        mapped = _mapped_path(mount_point, _suffix(logical, mount_root))
        if scopes[index] != (kind, logical, mapped, "searchable"):
            raise ValueError(f"scope {index} differs from the exact task-to-job chain")
    job_scope_path = _mapped_path(mount_point, _suffix(job_boundary, mount_root))

    metric_names = ("memory.current", "memory.max", "memory.events", "memory.events.local")
    expected_metric_keys = {
        (index, name) for index in range(len(expected_logical)) for name in metric_names
    }
    if set(metrics) != expected_metric_keys:
        raise ValueError("per-scope cgroup metric identities are incomplete or changed")
    valid_counter_scopes = set(range(len(expected_logical)))
    valid_counter_sources = {"memory.events", "memory.events.local"}
    if any(
        index not in valid_counter_scopes or source not in valid_counter_sources
        for index, source, _ in counters
    ):
        raise ValueError("per-scope cgroup counter identity is invalid")
    scope_diagnostics: list[dict[str, Any]] = []
    for index in range(len(expected_logical)):
        current_state, current_text = metrics[(index, "memory.current")]
        if current_state not in _SCALAR_STATES or current_state == "readable_unlimited":
            raise ValueError(f"scope {index} memory.current state is invalid")
        if current_state == "readable_integer":
            current_value: int | None = _uint(
                current_text, label=f"scope {index} memory.current"
            )
        else:
            if current_text != "-":
                raise ValueError(f"scope {index} unavailable memory.current needs '-' sentinel")
            current_value = None
        max_state, max_text = metrics[(index, "memory.max")]
        if max_state == "readable_integer":
            max_value: int | None = _uint(
                max_text, label=f"scope {index} memory.max", positive=True
            )
        elif max_state == "readable_unlimited" and max_text == "max":
            max_value = None
        elif max_state in _SCALAR_STATES - {"readable_integer", "readable_unlimited"}:
            if max_text != "-":
                raise ValueError(f"scope {index} unavailable memory.max needs '-' sentinel")
            max_value = None
        else:
            raise ValueError(f"scope {index} memory.max state is invalid")
        source_counters: dict[str, dict[str, int]] = {}
        for name in ("memory.events", "memory.events.local"):
            state, diagnostic_value = metrics[(index, name)]
            if state not in _EVENT_STATES or diagnostic_value != "-":
                raise ValueError(f"scope {index} {name} diagnostic state is invalid")
            observed = {
                key: value
                for (scope_index, source, key), value in counters.items()
                if scope_index == index and source == name
            }
            if state == "readable_flat_keys":
                if not REQUIRED_EVENT_KEYS.issubset(observed):
                    raise ValueError(f"scope {index} {name} counters are incomplete")
            elif observed:
                raise ValueError(f"scope {index} unavailable {name} has counters")
            source_counters[name] = dict(sorted(observed.items()))
        scope_diagnostics.append(
            {
                "index": index,
                "kind": scopes[index][0],
                "logical_path": scopes[index][1],
                "mapped_path": scopes[index][2],
                "memory_current_bytes": current_value,
                "memory_max_state": max_state,
                "memory_max_bytes": max_value,
                "memory_events_state": metrics[(index, "memory.events")][0],
                "memory_events": source_counters["memory.events"],
                "memory_events_local_state": metrics[(index, "memory.events.local")][0],
                "memory_events_local": source_counters["memory.events.local"],
            }
        )

    if set(snapshots) != {"before", "after"}:
        raise ValueError("memory.max before/after snapshots are incomplete")
    maximums: dict[str, int] = {}
    for phase in ("before", "after"):
        state, text_value = snapshots[phase]
        if state != "readable_integer":
            raise ValueError(f"memory.max {phase} is not a finite integer")
        maximums[phase] = _uint(
            text_value, label=f"memory.max {phase}", positive=True
        )
    if maximums["before"] != maximums["after"]:
        raise ValueError("memory.max changed during the sampled workload")
    job_scope_index = len(expected_logical) - 1
    if scope_diagnostics[job_scope_index]["memory_current_bytes"] is None:
        raise ValueError("job-scope memory.current diagnostic is not readable")
    if scope_diagnostics[job_scope_index]["memory_max_state"] != "readable_integer":
        raise ValueError("job-scope memory.max diagnostic is not finite")
    if scope_diagnostics[job_scope_index]["memory_max_bytes"] != maximums["before"]:
        raise ValueError("job-scope memory.max diagnostic differs from the raw snapshot")

    if set(events["before"]) != set(events["after"]):
        raise ValueError("memory.events before/after key sets differ")
    if not REQUIRED_EVENT_KEYS.issubset(events["before"]):
        raise ValueError("hierarchical memory.events map is incomplete")
    event_deltas: dict[str, int] = {}
    for key in sorted(events["before"]):
        before = events["before"][key]
        after = events["after"][key]
        if after < before:
            raise ValueError(f"memory.events counter {key} decreased")
        event_deltas[key] = after - before
    if any(event_deltas[key] != 0 for key in CRITICAL_EVENT_KEYS):
        raise ValueError("memory.events recorded a new max or OOM event")
    if set(local_events["before"]) != set(local_events["after"]):
        raise ValueError("memory.events.local before/after key sets differ")
    if not REQUIRED_EVENT_KEYS.issubset(local_events["before"]):
        raise ValueError("diagnostic memory.events.local map is incomplete")
    local_event_deltas: dict[str, int] = {}
    for key in sorted(local_events["before"]):
        before = local_events["before"][key]
        after = local_events["after"][key]
        if after < before:
            raise ValueError(f"memory.events.local counter {key} decreased")
        local_event_deltas[key] = after - before
    if scope_diagnostics[job_scope_index]["memory_events_state"] != "readable_flat_keys":
        raise ValueError("job-scope memory.events diagnostic is unavailable")
    if scope_diagnostics[job_scope_index]["memory_events"] != events["before"]:
        raise ValueError("job-scope memory.events diagnostic differs from the before snapshot")
    if (
        scope_diagnostics[job_scope_index]["memory_events_local_state"]
        != "readable_flat_keys"
    ):
        raise ValueError("job-scope memory.events.local diagnostic is unavailable")
    if scope_diagnostics[job_scope_index]["memory_events_local"] != local_events["before"]:
        raise ValueError(
            "job-scope memory.events.local diagnostic differs from the before snapshot"
        )

    if (
        sorted(samples) != list(range(len(samples)))
        or sample_row_order != list(range(len(samples)))
        or len(samples) < 2
    ):
        raise ValueError("sample indices are incomplete or noncontiguous")
    ordered_samples = [samples[index] for index in range(len(samples))]
    timestamps = [item[0] for item in ordered_samples]
    values = [item[1] for item in ordered_samples]
    gaps = [right - left for left, right in zip(timestamps, timestamps[1:])]
    if any(gap <= 0 for gap in gaps):
        raise ValueError("sample timestamps are not strictly increasing")
    maximum_gap = max(gaps)
    if maximum_gap > MAX_ADJACENT_GAP_NS:
        raise ValueError("sample trace exceeds the registered maximum adjacent gap")
    high_water = max(values)
    if high_water <= 0 or high_water > maximums["before"]:
        raise ValueError("sampled current high-water is outside the job-scope hard limit")

    if (
        sorted(lifecycles) != list(range(len(LIFECYCLE_MARKERS)))
        or lifecycle_row_order != list(range(len(LIFECYCLE_MARKERS)))
    ):
        raise ValueError("lifecycle indices are incomplete or noncontiguous")
    ordered_lifecycle = [lifecycles[index] for index in range(len(LIFECYCLE_MARKERS))]
    if tuple(item[0] for item in ordered_lifecycle) != LIFECYCLE_MARKERS:
        raise ValueError("lifecycle marker order changed")
    lifecycle = {name: timestamp for name, timestamp in ordered_lifecycle}
    lifecycle_times = [item[1] for item in ordered_lifecycle]
    if any(right < left for left, right in zip(lifecycle_times, lifecycle_times[1:])):
        raise ValueError("lifecycle timestamps are out of order")
    if not timestamps[0] <= lifecycle["monitor_ready"] <= lifecycle["policy_launch"]:
        raise ValueError("first sample does not cover monitor-ready through policy launch")
    if not (
        lifecycle["workload_cleanup_complete"]
        <= lifecycle["monitor_stop_observed"]
        <= timestamps[-1]
    ):
        raise ValueError("final sample does not cover workload cleanup and stop observation")
    lifecycle_positions = [
        positions[("lifecycle", str(index))] for index in range(len(LIFECYCLE_MARKERS))
    ]
    if lifecycle_positions != sorted(lifecycle_positions):
        raise ValueError("lifecycle rows are out of order")

    if set(capabilities) != {"job_memory_peak"}:
        raise ValueError("native memory.peak capability record changed")
    peak_state, peak_text = capabilities["job_memory_peak"]
    if peak_state == "missing":
        if peak_text != "-":
            raise ValueError("missing native memory.peak has a value")
        peak_value: int | None = None
    elif peak_state == "readable_integer":
        peak_value = _uint(peak_text, label="native memory.peak", positive=True)
    else:
        raise ValueError("native memory.peak state is invalid")

    if set(summaries) != set(SUMMARY_KEYS) or summary_row_order != list(SUMMARY_KEYS):
        raise ValueError("trace summary keys changed")
    expected_summary = {
        "job_scope_path": job_scope_path,
        "job_memory_max_before_bytes": str(maximums["before"]),
        "job_memory_max_after_bytes": str(maximums["after"]),
        "sample_count_observed": str(len(samples)),
        "sample_interval_requested_seconds": "0.1",
        "first_sample_monotonic_ns": str(timestamps[0]),
        "last_sample_monotonic_ns": str(timestamps[-1]),
        "sample_max_gap_ns": str(maximum_gap),
        "sampled_memory_current_high_water_bytes": str(high_water),
        "memory_events_max_delta": str(event_deltas["max"]),
        "memory_events_oom_delta": str(event_deltas["oom"]),
        "memory_events_oom_kill_delta": str(event_deltas["oom_kill"]),
        "native_memory_peak_state": peak_state,
        "native_memory_peak_value": "-" if peak_value is None else str(peak_value),
        "first_sample_no_later_than_policy_launch": "true",
        "last_sample_no_earlier_than_workload_cleanup": "true",
        "host_cgroup_sampled_current_is_lower_bound_not_peak": "true",
        "outcome": "full_lifetime_sampled_current_supported",
    }
    if summaries != expected_summary:
        raise ValueError("trace summary differs from independent raw recomputation")
    first_sample_position = positions[("sample", "0")]
    last_sample_position = positions[("sample", str(len(samples) - 1))]
    if positions[("snapshot", "before")] >= first_sample_position:
        raise ValueError("pre-workload memory.max was recorded after sampling began")
    if any(position >= first_sample_position for position in event_positions["before"]):
        raise ValueError("a pre-workload memory.events row was recorded after sampling began")
    if any(
        position >= first_sample_position
        for position in local_event_positions["before"]
    ):
        raise ValueError(
            "a pre-workload memory.events.local row was recorded after sampling began"
        )
    if positions[("snapshot", "after")] <= last_sample_position:
        raise ValueError("post-workload memory.max was recorded before the final sample")
    if any(position <= last_sample_position for position in event_positions["after"]):
        raise ValueError("a post-workload memory.events row preceded the final sample")
    if any(
        position <= last_sample_position for position in local_event_positions["after"]
    ):
        raise ValueError(
            "a post-workload memory.events.local row preceded the final sample"
        )
    if any(position <= last_sample_position for position in lifecycle_positions):
        raise ValueError("lifecycle rows were written before the final sample")
    capability_position = positions[("capability", "job_memory_peak")]
    if capability_position <= last_sample_position:
        raise ValueError("native memory.peak capability preceded the final sample")
    first_summary_position = min(positions[("summary", key)] for key in SUMMARY_KEYS)
    last_raw_position = max(
        positions[("snapshot", "after")],
        capability_position,
        *event_positions["after"],
        *local_event_positions["after"],
        *lifecycle_positions,
    )
    if first_summary_position <= last_raw_position:
        raise ValueError("trace summary was written before raw telemetry sealed")

    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "artifact_role": SUMMARY_ARTIFACT_ROLE,
        "raw_trace_path": str(trace_path),
        "raw_trace_sha256": _sha256(raw),
        "cgroup_version": 2,
        "slurm_array_job_id": expected_job_text,
        "slurm_array_task_id": 0,
        "membership_path": membership,
        "mount_root": mount_root,
        "mount_point": mount_point,
        "job_boundary_path": job_boundary,
        "job_scope_path": job_scope_path,
        "kernel_osrelease": records["kernel_osrelease"],
        "scope_diagnostics": scope_diagnostics,
        "memory_events_hierarchy_state": "hierarchical",
        "sample_interval_requested_ms": REQUESTED_INTERVAL_MS,
        "sample_count": len(samples),
        "first_sample_monotonic_ns": timestamps[0],
        "last_sample_monotonic_ns": timestamps[-1],
        "maximum_adjacent_gap_ns": maximum_gap,
        "registered_maximum_adjacent_gap_ns": MAX_ADJACENT_GAP_NS,
        "host_cgroup_sampled_current_high_water_bytes": high_water,
        "host_cgroup_sampled_current_is_lower_bound_not_peak": True,
        "memory_max_before_bytes": maximums["before"],
        "memory_max_after_bytes": maximums["after"],
        "memory_max_unchanged": True,
        "memory_max_job_scope_hard_limit_bytes": maximums["before"],
        "memory_max_is_usage": False,
        "memory_max_is_claimed_tightest_hierarchy_limit": False,
        "memory_events_before": dict(sorted(events["before"].items())),
        "memory_events_after": dict(sorted(events["after"].items())),
        "memory_events_deltas": dict(sorted(event_deltas.items())),
        "memory_events_local_before": dict(sorted(local_events["before"].items())),
        "memory_events_local_after": dict(sorted(local_events["after"].items())),
        "memory_events_local_deltas": dict(sorted(local_event_deltas.items())),
        "native_memory_peak_state": peak_state,
        "native_memory_peak_value_bytes": peak_value,
        "lifecycle_monotonic_ns": lifecycle,
        "first_sample_no_later_than_policy_launch": True,
        "last_sample_no_earlier_than_workload_cleanup": True,
        "full_lifetime_window_passed": True,
        "contract_passed": True,
    }


def validate_full_lifetime_telemetry(
    path: str | Path, *, expected_job_id: str | int
) -> tuple[dict[str, Any] | None, list[str]]:
    """Return a normalized summary and errors without raising to callers."""

    try:
        return parse_full_lifetime_telemetry(path, expected_job_id=expected_job_id), []
    except (OSError, UnicodeError, ValueError) as error:
        return None, [str(error)]


__all__ = [
    "MAX_ADJACENT_GAP_NS",
    "TRACE_ARTIFACT_ROLE",
    "parse_full_lifetime_telemetry",
    "validate_full_lifetime_telemetry",
]
