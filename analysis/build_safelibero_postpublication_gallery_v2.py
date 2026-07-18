#!/usr/bin/env python3
"""Build an evidence-rich paired gallery from immutable analysis-v2 rows.

The gallery consumes only the compact, hash-bound v2 case ledger.  It does
not reinterpret raw simulator artifacts, edit videos, or invent clearance.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

try:
    from analysis import aggregate_safelibero_aegis as aggregate
    from analysis import build_aegis_failure_report as failures
except ImportError:  # pragma: no cover - direct script fallback
    import aggregate_safelibero_aegis as aggregate  # type: ignore[no-redef]
    import build_aegis_failure_report as failures  # type: ignore[no-redef]


GALLERY_SCHEMA = "vlsa_table1_postpublication_gallery_v2.v1"


class GalleryV2Error(RuntimeError):
    """Raised when compact v2 evidence is incomplete or inconsistent."""


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GalleryV2Error(f"{label} must be an object")
    return value


def _sha(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise GalleryV2Error(f"{label} must be lowercase SHA-256")
    return value


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise GalleryV2Error(f"{label} is missing or symlinked")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GalleryV2Error(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise GalleryV2Error(f"{label} must be an object")
    return value


def _verify_payload(
    value: Mapping[str, Any],
    *,
    field: str,
    label: str,
) -> str:
    expected = _sha(value.get(field), label=f"{label}/{field}")
    payload = dict(value)
    payload.pop(field, None)
    if (
        aggregate.sha256_bytes(aggregate.canonical_json_bytes(payload))
        != expected
    ):
        raise GalleryV2Error(f"{label} payload hash differs")
    return expected


def load_cases_v2(
    path: Path,
    *,
    summary: Mapping[str, Any],
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Load and bind all compact rows to the v2 summary and report."""

    if path.is_symlink() or not path.is_file():
        raise GalleryV2Error("analysis-v2 case ledger is unavailable")
    population = _mapping(
        summary.get("population"), label="analysis-v2 population"
    )
    expected_cases = population.get("cases")
    if type(expected_cases) is not int or expected_cases <= 0:
        raise GalleryV2Error("analysis-v2 case denominator is invalid")
    records: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    with path.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            if not raw_line.strip():
                raise GalleryV2Error("case ledger contains a blank row")
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise GalleryV2Error(
                    f"case row {line_number} is not valid JSON"
                ) from error
            if (
                not isinstance(row, dict)
                or row.get("schema_version") != failures.CASE_SCHEMA_V2
            ):
                raise GalleryV2Error(
                    f"case row {line_number} has the wrong schema"
                )
            case_id = row.get("case_id")
            if (
                not isinstance(case_id, str)
                or not case_id
                or case_id in case_ids
            ):
                raise GalleryV2Error("case identities are not unique")
            case_ids.add(case_id)
            _verify_payload(
                row,
                field="record_payload_sha256",
                label=f"case/{case_id}",
            )
            records.append(row)
    if len(records) != expected_cases:
        raise GalleryV2Error(
            f"case ledger has {len(records)} rows, expected {expected_cases}"
        )
    records.sort(key=lambda row: int(row["case_ordinal"]))
    case_ledger = [
        {
            "case_id": row["case_id"],
            "record_payload_sha256": row["record_payload_sha256"],
        }
        for row in records
    ]
    if (
        aggregate.sha256_bytes(
            aggregate.canonical_json_bytes(case_ledger)
        )
        != report.get("case_record_ledger_sha256")
    ):
        raise GalleryV2Error("case ledger differs from the v2 report")
    result_ledger = sorted(
        (
            {
                "case_id": row["case_id"],
                "arm": arm,
                "result_payload_sha256": _sha(
                    row["outcomes"][arm].get("result_payload_sha256"),
                    label=f"{row['case_id']}/{arm}/result payload",
                ),
            }
            for row in records
            for arm in (failures.BASELINE_ARM, failures.AEGIS_ARM)
        ),
        key=lambda item: (item["case_id"], item["arm"]),
    )
    if (
        aggregate.sha256_bytes(
            aggregate.canonical_json_bytes(result_ledger)
        )
        != summary.get("accepted_result_payloads_sha256")
    ):
        raise GalleryV2Error(
            "case rows differ from the immutable accepted-result ledger"
        )
    return records


def load_bound_inputs(
    *,
    summary_path: Path,
    report_path: Path,
    cases_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    summary = _load_json(summary_path, label="analysis-v2 summary")
    report = _load_json(report_path, label="analysis-v2 report")
    if (
        summary.get("schema_version") != aggregate.OUTPUT_SCHEMA_V2
        or summary.get("status")
        != "complete_postpublication_analysis_v2"
    ):
        raise GalleryV2Error("analysis-v2 summary is not complete")
    if (
        report.get("schema_version") != failures.REPORT_SCHEMA_V2
        or report.get("status")
        != "complete_postpublication_failure_analysis_v2"
    ):
        raise GalleryV2Error("analysis-v2 report is not complete")
    _verify_payload(
        report,
        field="report_payload_sha256",
        label="analysis-v2 report",
    )
    source = _mapping(
        report.get("source"), label="analysis-v2 report source"
    )
    source_v1 = _mapping(
        summary.get("source_v1"), label="analysis-v2 source-v1 binding"
    )
    if (
        _sha(
            source.get("population_summary_v2_sha256"),
            label="report summary SHA",
        )
        != aggregate.sha256_path(summary_path)
        or source.get("accepted_result_payloads_sha256")
        != summary.get("accepted_result_payloads_sha256")
        or _sha(
            source.get("v1_publication_receipt_sha256"),
            label="report v1 publication receipt SHA",
        )
        != _sha(
            source_v1.get("v1_publication_receipt_sha256"),
            label="summary v1 publication receipt SHA",
        )
        or report.get("claim_scope") != summary.get("claim_scope")
    ):
        raise GalleryV2Error(
            "analysis-v2 report is not bound to the supplied summary"
        )
    records = load_cases_v2(
        cases_path, summary=summary, report=report
    )
    counts = _mapping(report.get("counts"), label="analysis-v2 counts")
    if (
        counts.get("case_count") != len(records)
        or counts.get("video_count") != len(records) * 2
        or counts.get("all_videos_hash_verified") is not True
    ):
        raise GalleryV2Error("analysis-v2 report/video denominator changed")
    return summary, report, records


def _text(value: Any) -> str:
    return html.escape(str(value))


def _attr(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _option(value: str, label: str | None = None) -> str:
    return (
        f'<option value="{_attr(value)}">'
        f"{_text(value if label is None else label)}</option>"
    )


def _video_href(
    video: Mapping[str, Any],
    *,
    output_root: Path,
) -> str:
    if video.get("hash_verified") is not True:
        raise GalleryV2Error("gallery video is not hash verified")
    raw = video.get("absolute_path")
    if not isinstance(raw, str) or not raw:
        raise GalleryV2Error("gallery video path is missing")
    path = Path(raw)
    if not path.is_absolute() or not path.is_file():
        raise GalleryV2Error(f"gallery video is unavailable: {raw}")
    return quote(
        Path(
            os.path.relpath(path.resolve(), start=output_root.resolve())
        ).as_posix(),
        safe="/",
    )


def _outcome_label(outcome: Mapping[str, Any]) -> tuple[str, str]:
    car = "collision" if outcome["paper_collision"] else "safe"
    task = "success" if outcome["task_success"] else "failure"
    return car, task


def _render_arm(
    *,
    arm: str,
    outcome: Mapping[str, Any],
    video: Mapping[str, Any],
    output_root: Path,
) -> str:
    car, task = _outcome_label(outcome)
    href = _video_href(video, output_root=output_root)
    contact = bool(outcome["sampled_robot_active_obstacle_contact"])
    disagreement = (car == "collision") != contact
    return f"""
      <section class="arm-panel" data-arm="{_attr(arm)}"
        data-car="{_attr(car)}" data-task="{_attr(task)}"
        data-contact-disagreement="{_attr('yes' if disagreement else 'no')}">
        <h3>{_text(arm)}</h3>
        <p><span class="badge {car}">CAR: {_text(car)}</span>
          <span class="badge {task}">TSR: {_text(task)}</span>
          <span class="badge">sampled robot contact:
            {_text('yes' if contact else 'no')}</span></p>
        <video controls preload="metadata">
          <source src="{_attr(href)}" type="video/mp4">
        </video>
        <p><a href="{_attr(href)}">Open exact MP4</a> · ETS
          {_text(outcome['legacy_ets_steps'])} · executed actions
          {_text(outcome['executed_action_count'])}</p>
      </section>
    """


def _render_case(
    record: Mapping[str, Any],
    *,
    output_root: Path,
) -> str:
    outcomes = _mapping(record["outcomes"], label="case outcomes")
    videos = _mapping(record["videos"], label="case videos")
    failure = _mapping(
        record["failure_analysis"], label="case failure analysis"
    )
    progress = _mapping(
        record["paired_goal_progress"], label="case goal progress"
    )
    delta = _mapping(
        progress["aegis_minus_pi05"], label="case goal delta"
    )
    intervention = _mapping(
        record["aegis_diagnostics"]["intervention"],
        label="case intervention",
    )
    baseline = _mapping(
        outcomes[failures.BASELINE_ARM], label="baseline outcome"
    )
    aegis = _mapping(
        outcomes[failures.AEGIS_ARM], label="AEGIS outcome"
    )
    baseline_car, baseline_task = _outcome_label(baseline)
    aegis_car, aegis_task = _outcome_label(aegis)
    primary = str(failure["primary_car_failure_class"])
    no_points = str(failure["no_usable_points_observed_subclass"])
    disagreement = any(
        bool(outcomes[arm]["paper_collision"])
        != bool(
            outcomes[arm]["sampled_robot_active_obstacle_contact"]
        )
        for arm in (failures.BASELINE_ARM, failures.AEGIS_ARM)
    )
    observed = "".join(
        f"<li>{_text(value)}</li>"
        for value in failure["observed_evidence_tags"]
    )
    hypotheses = "".join(
        f"<li>{_text(value)}</li>"
        for value in failure["causal_hypotheses"]
    )
    panels = "".join(
        _render_arm(
            arm=arm,
            outcome=_mapping(outcomes[arm], label=f"{arm}/outcome"),
            video=_mapping(videos[arm], label=f"{arm}/video"),
            output_root=output_root,
        )
        for arm in (failures.BASELINE_ARM, failures.AEGIS_ARM)
    )
    task_key = (
        f"{record['suite']}|{record['safety_level']}|"
        f"{record['logical_task_index']}|{record['task_name']}"
    )
    return f"""
    <article class="case-card" data-suite="{_attr(record['suite'])}"
      data-level="{_attr(record['safety_level'])}"
      data-task="{_attr(task_key)}"
      data-label="{_attr(record['frozen_obstacle_label'])}"
      data-baseline-car="{_attr(baseline_car)}"
      data-aegis-car="{_attr(aegis_car)}"
      data-aegis-task="{_attr(aegis_task)}"
      data-contact-disagreement="{_attr('yes' if disagreement else 'no')}"
      data-primary="{_attr(primary)}"
      data-no-points="{_attr(no_points)}">
      <header><h2>{_text(record['case_id'])}</h2>
        <p>{_text(record['suite'])} · level
          {_text(record['safety_level'])} · task
          {_text(record['logical_task_index'])} ·
          {_text(record['frozen_obstacle_label'])}</p></header>
      <div class="paired-grid">{panels}</div>
      <details><summary>Validated analysis-v2 evidence</summary>
        <p><strong>Primary observed CAR class:</strong>
          <code>{_text(primary)}</code></p>
        <p><strong>No-usable-points subclass:</strong>
          <code>{_text(no_points)}</code></p>
        <p><strong>Paired transitions:</strong>
          {_text(record['paired_transition'])}</p>
        <p><strong>AEGIS − pi0.5 goal progress:</strong>
          final fraction {_text(delta['final_fraction'])},
          maximum fraction {_text(delta['maximum_fraction'])},
          regressions {_text(delta['regression_count'])}.</p>
        <p><strong>AEGIS interventions:</strong>
          {_text(intervention['intervention_count'])} /
          {_text(intervention['eligible_steps'])} eligible actions;
          correction L2 sum
          {_text(intervention['correction_l2_sum'])}.</p>
        <div class="evidence-columns"><section><h4>Observed</h4>
          <ul>{observed}</ul></section><section><h4>Hypotheses—not
          established causes</h4><ul>{hypotheses}</ul></section></div>
      </details>
    </article>
    """


def render_gallery_v2(
    *,
    summary: Mapping[str, Any],
    report: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    output_root: Path,
) -> str:
    scope = _mapping(summary["claim_scope"], label="gallery claim scope")
    values = {
        "suite": sorted({str(row["suite"]) for row in records}),
        "level": sorted({str(row["safety_level"]) for row in records}),
        "task": sorted(
            {
                (
                    f"{row['suite']}|{row['safety_level']}|"
                    f"{row['logical_task_index']}|{row['task_name']}"
                )
                for row in records
            }
        ),
        "label": sorted(
            {str(row["frozen_obstacle_label"]) for row in records}
        ),
        "primary": sorted(
            {
                str(
                    row["failure_analysis"][
                        "primary_car_failure_class"
                    ]
                )
                for row in records
            }
        ),
        "no-points": sorted(
            {
                str(
                    row["failure_analysis"][
                        "no_usable_points_observed_subclass"
                    ]
                )
                for row in records
            }
        ),
    }
    filters = [
        ("suite", "Suite", values["suite"]),
        ("level", "Level", values["level"]),
        ("task", "Task", values["task"]),
        ("label", "Frozen obstacle label", values["label"]),
        ("baseline-car", "pi0.5 CAR", ["safe", "collision"]),
        ("aegis-car", "AEGIS CAR", ["safe", "collision"]),
        ("aegis-task", "AEGIS TSR", ["success", "failure"]),
        ("contact-disagreement", "CAR/contact disagreement", ["yes", "no"]),
        ("primary", "Primary observed class", values["primary"]),
        ("no-points", "No-usable-points subclass", values["no-points"]),
    ]
    filter_html = "".join(
        (
            f'<label>{_text(label)}<select id="{_attr(key)}-filter">'
            + _option("", "All")
            + "".join(_option(value) for value in options)
            + "</select></label>"
        )
        for key, label, options in filters
    )
    cards = "".join(
        _render_case(record, output_root=output_root)
        for record in records
    )
    filter_ids = json.dumps([key for key, _, _ in filters])
    return f"""<!doctype html>
<html lang="en" data-gallery-schema="{GALLERY_SCHEMA}">
<head><meta charset="utf-8"><meta name="viewport"
  content="width=device-width, initial-scale=1">
<title>SafeLIBERO post-publication analysis v2</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#eef2f7;color:#17202a}}
main{{max-width:1500px;margin:auto;padding:22px}} .scope,.filters,.case-card{{
background:white;border:1px solid #d7dce3;border-radius:12px;padding:14px;
margin:14px 0}} .filters{{display:flex;gap:9px;flex-wrap:wrap;position:sticky;
top:0;z-index:2}} label{{display:grid;font-size:12px;color:#566273}}
select,button{{max-width:260px;min-height:34px}} .paired-grid{{display:grid;
grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}} .arm-panel{{
background:#f6f8fb;border-radius:9px;padding:10px;min-width:0}} video{{
width:100%;max-height:420px;background:#111}} .badge{{display:inline-block;
padding:3px 7px;border-radius:999px;background:#e5e9ef;font-size:12px}}
.safe,.success{{background:#d9f3e4}} .collision,.failure{{background:#ffe0dc}}
.evidence-columns{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
[hidden]{{display:none!important}} @media(max-width:850px){{.paired-grid,
.evidence-columns{{grid-template-columns:1fr}}.filters{{position:static}}}}
</style></head><body><main>
<h1>SafeLIBERO post-publication analysis v2</h1>
<section class="scope"><p><strong>Baseline:</strong>
{_text(scope['baseline_method_label'])}. <strong>Compared method:</strong>
{_text(scope['aegis_method_label'])}.</p>
<p>{_text(scope['population_scope'])}. OpenVLA-OFT is not included; the
paper semantic selector is not reproduced. No continuous minimum-clearance
signal exists, so this gallery makes no clearance claim.</p>
<p>Cases: {_text(summary['population']['cases'])}; videos:
{_text(report['counts']['video_count'])}. Observed evidence and causal
hypotheses are displayed separately.</p></section>
<section class="filters">{filter_html}<button id="reset">Reset</button>
<output id="visible"></output></section>
<section id="gallery">{cards}</section>
</main><script>
(() => {{
const ids={filter_ids}; const cards=[...document.querySelectorAll(".case-card")];
const fields=Object.fromEntries(ids.map(id=>[id,document.getElementById(id+"-filter")]));
const output=document.getElementById("visible");
function apply(){{let visible=0;for(const card of cards){{
const match=ids.every(id=>!fields[id].value||
card.getAttribute("data-"+id)===fields[id].value);
card.hidden=!match;if(match)visible+=1;}}output.value=visible+" / "+cards.length+" cases";}}
Object.values(fields).forEach(field=>field.addEventListener("change",apply));
document.getElementById("reset").addEventListener("click",()=>{{
Object.values(fields).forEach(field=>field.value="");apply();}});apply();
}})();
</script></body></html>
"""


def build_gallery_v2(
    *,
    summary: Mapping[str, Any],
    report: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    output_root: Path,
) -> tuple[str, dict[str, Any]]:
    if not records:
        raise GalleryV2Error("gallery requires at least one case")
    document = render_gallery_v2(
        summary=summary,
        report=report,
        records=records,
        output_root=output_root,
    )
    metadata = {
        "schema_version": GALLERY_SCHEMA,
        "cases": len(records),
        "videos": len(records) * 2,
        "case_record_ledger_sha256": report[
            "case_record_ledger_sha256"
        ],
        "accepted_result_payloads_sha256": summary[
            "accepted_result_payloads_sha256"
        ],
        "claim_scope": summary["claim_scope"],
    }
    return document, metadata


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the immutable enriched analysis-v2 gallery"
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary, report, records = load_bound_inputs(
        summary_path=args.summary.resolve(),
        report_path=args.report.resolve(),
        cases_path=args.cases.resolve(),
    )
    output = args.output.resolve()
    document, metadata = build_gallery_v2(
        summary=summary,
        report=report,
        records=records,
        output_root=output.parent,
    )
    atomic_write_text(output, document)
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
