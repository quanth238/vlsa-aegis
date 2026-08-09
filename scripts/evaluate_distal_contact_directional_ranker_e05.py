#!/usr/bin/env python3
"""Train paired directional contact risk and test it against matched random."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

from main.multilink_ellipsoid.contact_directional_ranker import (
    burden_key,
    empirical_burden_p,
    symmetric_directional_pairs,
)
from main.multilink_ellipsoid.contact_gradient_sign_audit import contact_burden
from main.multilink_ellipsoid.gradient_random_control import (
    sample_feasible_unit_directions,
)
from scripts.audit_distal_contact_gradient_sign_e05 import _predict
from scripts.collect_distal_boundary_generalization_moka10 import (
    _restore_env,
    _snapshot_env,
)
from scripts.evaluate_distal_contact_ranker_e05 import (
    _build_model,
    _risk_metrics,
    _successful_query_action,
)
from scripts.evaluate_distal_execution_margin_nn_e05 import _build_pair, _source_action
from scripts.replay_distal_three_ellipsoid_multicbf import (
    CASE_ID,
    _atomic_write,
    _file_sha256,
    _git_identity,
    _load,
    _require,
)


SCHEMA = "vlsa_distal_contact_directional_ranker_e05.v1"
RESULT_SCHEMA = "vlsa_distal_contact_directional_ranker_e05_result.v1"


def _hash_without(value, key):
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(config.get("schema_version") == SCHEMA, "directional-ranker config schema differs")
    _require(config.get("case_id") == CASE_ID, "directional-ranker case differs")
    groups = config.get("state_groups", {})
    _require(groups.get("train_steps") == [184, 185, 186, 188, 189, 191], "directional train states differ")
    _require(groups.get("validation_steps") == [192], "directional validation state differs")
    _require(groups.get("test_steps") == [187, 190], "directional test states differ")
    _require(config.get("matched_random_test", {}).get("radius_action") == 0.1, "directional test radius differs")
    return config


def _train(dataset, pairs, config, model_path):
    import numpy as np
    import torch

    records = dataset["records"]
    features = np.asarray([item["feature_vector"] for item in records], dtype=np.float64)
    labels = np.asarray([item["link_contact_labels"] for item in records], dtype=np.float64)
    splits = np.asarray([item["split"] for item in records])
    record_positions = {int(item["record_index"]): index for index, item in enumerate(records)}
    _require(len(record_positions) == len(records), "directional record indexes are not unique")
    train_mask = splits == "train"
    settings = config["training"]
    mean = np.mean(features[train_mask], axis=0)
    deviation = np.maximum(
        np.std(features[train_mask], axis=0),
        float(settings["minimum_standard_deviation"]),
    )
    normalized = (features - mean) / deviation
    seed = int(settings["seed"])
    torch.manual_seed(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    model = _build_model(torch, features.shape[1], config["network"]["hidden_widths"]).to(
        device=device, dtype=torch.float64
    )
    x = torch.as_tensor(normalized, dtype=torch.float64, device=device)
    y = torch.as_tensor(labels, dtype=torch.float64, device=device)
    positive = np.sum(labels[train_mask], axis=0)
    negative = int(np.count_nonzero(train_mask)) - positive
    positive_weight = np.where(positive > 0, negative / np.maximum(positive, 1.0), 1.0)
    bce = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.as_tensor(positive_weight, dtype=torch.float64, device=device)
    )
    split_positions = {
        split: np.flatnonzero(splits == split) for split in ("train", "validation", "test")
    }
    local_positions = {
        split: {int(global_index): local for local, global_index in enumerate(indexes)}
        for split, indexes in split_positions.items()
    }
    informative_pairs = {
        split: [item for item in pairs if item["split"] == split and item["informative"]]
        for split in ("train", "validation", "test")
    }
    pair_local = {}
    for split, selected in informative_pairs.items():
        local = local_positions[split]
        pair_local[split] = [(
            local[record_positions[int(item["better_record_index"])]],
            local[record_positions[int(item["worse_record_index"])]],
        ) for item in selected]
    _require(pair_local["train"] and pair_local["validation"], "directional training lacks informative train/validation pairs")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    margin = float(settings["directional_margin_probability"])
    weight = float(settings["directional_loss_weight"])

    def loss_for(split):
        indexes = split_positions[split]
        logits = model(x[indexes])
        classification = bce(logits, y[indexes])
        risk = torch.max(torch.sigmoid(logits), dim=1).values
        local_pairs = pair_local[split]
        better = torch.as_tensor([item[0] for item in local_pairs], dtype=torch.long, device=device)
        worse = torch.as_tensor([item[1] for item in local_pairs], dtype=torch.long, device=device)
        directional = torch.relu(margin - (risk[worse] - risk[better])).mean()
        return classification + weight * directional, classification, directional

    best = None
    best_epoch = None
    best_validation = math.inf
    patience = 0
    trace = []
    training_started = time.perf_counter_ns()
    for epoch in range(int(settings["epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, classification, directional = loss_for("train")
        total.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_total, validation_classification, validation_directional = loss_for("validation")
        value = float(validation_total.cpu())
        if epoch == 0 or (epoch + 1) % 100 == 0:
            trace.append({
                "epoch": int(epoch),
                "train_total": float(total.detach().cpu()),
                "train_classification": float(classification.detach().cpu()),
                "train_directional": float(directional.detach().cpu()),
                "validation_total": value,
                "validation_classification": float(validation_classification.cpu()),
                "validation_directional": float(validation_directional.cpu()),
            })
        if value < best_validation - 1e-12:
            best_validation = value
            best_epoch = epoch
            best = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= int(settings["early_stopping_patience"]):
            break
    _require(best is not None, "directional training produced no model")
    model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(x)).detach().cpu().numpy()
    risks = np.max(probabilities, axis=1)

    def pair_metrics(split):
        selected = [item for item in pairs if item["split"] == split]
        informative = [item for item in selected if item["informative"]]
        correct = 0
        margins = []
        for item in informative:
            better = record_positions[int(item["better_record_index"])]
            worse = record_positions[int(item["worse_record_index"])]
            difference = float(risks[worse] - risks[better])
            correct += difference > 0.0
            margins.append(difference)
        return {
            "total_symmetric_pair_count": len(selected),
            "informative_pair_count": len(informative),
            "tie_pair_count": len(selected) - len(informative),
            "informative_pair_accuracy": None if not informative else correct / len(informative),
            "risk_ordering_margin_mean": None if not margins else float(np.mean(margins)),
            "risk_ordering_margin_minimum": None if not margins else float(np.min(margins)),
        }

    model_artifact = {
        "schema_version": "vlsa_distal_contact_directional_ranker_e05_model.v1",
        "model_state_dict": best,
        "feature_mean": mean,
        "feature_standard_deviation": deviation,
        "hidden_widths": list(config["network"]["hidden_widths"]),
        "protected_links": ["robot0_link5", "robot0_link6", "robot0_link7"],
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.parent / (".%s.%d.tmp" % (model_path.name, os.getpid()))
    torch.save(model_artifact, temporary)
    os.replace(temporary, model_path)
    audit = {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "epochs_run": int(epoch + 1),
        "best_validation_combined_loss": best_validation,
        "train_positive_count_by_link": positive.astype(int).tolist(),
        "train_negative_count_by_link": negative.astype(int).tolist(),
        "positive_weight_by_link": positive_weight.tolist(),
        "pair_metrics": {split: pair_metrics(split) for split in ("train", "validation", "test")},
        "classification_metrics": {
            split: _risk_metrics(
                [records[index] for index in split_positions[split]],
                risks[split_positions[split]],
                0.5,
            ) for split in ("train", "validation", "test")
        },
        "trace": trace,
        "training_device": "cpu_inside_H100_allocation",
        "torch_version": str(torch.__version__),
        "model_file_sha256": _file_sha256(model_path),
        "wall_seconds": (time.perf_counter_ns() - training_started) * 1e-9,
    }
    return model, model_artifact, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--successful-sitl", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--contact-dataset", type=Path, required=True)
    parser.add_argument("--contact-result", type=Path, required=True)
    parser.add_argument("--contact-validation", type=Path, required=True)
    parser.add_argument("--random-control-result", type=Path, required=True)
    parser.add_argument("--random-control-validation", type=Path, required=True)
    parser.add_argument("--sign-audit-result", type=Path, required=True)
    parser.add_argument("--sign-audit-validation", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()

    import numpy as np
    from main.evaluate_safelibero_aegis import (
        _runtime_imports, pairing_record, read_jsonl, validate_case_row,
    )
    from main.multilink_ellipsoid.execution_margin_nn import FEATURE_NAMES, feature_vector
    from main.multilink_ellipsoid.oracle_affine import SubstepEightConstraintProbe
    from main.multilink_ellipsoid.shadow import (
        MultilinkEllipsoidShadow, allocation_record, load_shadow_config,
    )

    config = load_config(args.config.resolve())
    identities = config["immutable_sources"]
    sources = {
        "archived_table1": args.archived.resolve(),
        "successful_sitl": args.successful_sitl.resolve(),
        "geometry_config": args.geometry_config.resolve(),
        "contact_dataset": args.contact_dataset.resolve(),
        "contact_result": args.contact_result.resolve(),
        "contact_validation": args.contact_validation.resolve(),
        "random_control_result": args.random_control_result.resolve(),
        "random_control_validation": args.random_control_validation.resolve(),
        "sign_audit_result": args.sign_audit_result.resolve(),
        "sign_audit_validation": args.sign_audit_validation.resolve(),
    }
    expected_files = {
        "archived_table1": identities["archived_table1_file_sha256"],
        "successful_sitl": identities["successful_sitl_file_sha256"],
        "geometry_config": identities["geometry_config_file_sha256"],
        "contact_dataset": identities["contact_dataset_file_sha256"],
        "contact_result": identities["contact_result_file_sha256"],
        "contact_validation": identities["contact_validation_file_sha256"],
        "random_control_result": identities["random_control_result_file_sha256"],
        "random_control_validation": identities["random_control_validation_file_sha256"],
        "sign_audit_result": identities["sign_audit_result_file_sha256"],
        "sign_audit_validation": identities["sign_audit_validation_file_sha256"],
    }
    for name, path in sources.items():
        _require(path.is_file() and not path.is_symlink(), "directional source missing: %s" % name)
        _require(_file_sha256(path) == expected_files[name], "directional source hash differs: %s" % name)
    archived = _load(sources["archived_table1"])
    successful = _load(sources["successful_sitl"])
    dataset = _load(sources["contact_dataset"])
    prior = _load(sources["contact_result"])
    prior_validation = _load(sources["contact_validation"])
    random_result = _load(sources["random_control_result"])
    random_validation = _load(sources["random_control_validation"])
    sign_result = _load(sources["sign_audit_result"])
    sign_validation = _load(sources["sign_audit_validation"])
    _require(successful.get("result_payload_sha256") == identities["successful_sitl_payload_sha256"], "directional successful payload differs")
    _require(dataset.get("dataset_payload_sha256") == identities["contact_dataset_payload_sha256"], "directional dataset payload differs")
    _require(prior.get("result_payload_sha256") == identities["contact_result_payload_sha256"], "directional prior payload differs")
    _require(random_result.get("result_payload_sha256") == identities["random_control_result_payload_sha256"], "directional random payload differs")
    _require(sign_result.get("result_payload_sha256") == identities["sign_audit_result_payload_sha256"], "directional sign payload differs")
    _require(prior_validation.get("status") == "passed" and random_validation.get("status") == "passed" and sign_validation.get("status") == "passed", "directional source validation differs")
    _require(sign_result.get("decision", {}).get("implementation_consistency_pass") is True, "directional sign prerequisite differs")
    _require(random_result.get("decision", {}).get("learned_gradient_beats_matched_random") is False, "directional random prerequisite differs")
    _require(dataset.get("source", {}).get("commit") == identities["contact_source_commit"], "directional dataset source differs")
    groups = config["state_groups"]
    for record in dataset["records"]:
        step = int(record["state_step"])
        expected_split = "train" if step in groups["train_steps"] else "validation" if step in groups["validation_steps"] else "test" if step in groups["test_steps"] else None
        _require(record["split"] == expected_split, "directional complete-state split differs")
    pair_settings = config["pair_construction"]
    pairs = symmetric_directional_pairs(
        dataset["records"],
        allowed_source_prefixes=pair_settings["allowed_candidate_source_prefixes"],
        maximum_radius_action=float(pair_settings["maximum_l2_radius_action"]),
        matching_tolerance=float(pair_settings["opposite_offset_matching_tolerance"]),
    )
    model, artifact, training = _train(dataset, pairs, config, args.model.resolve())

    rows = [row for row in read_jsonl(args.manifest.resolve()) if row.get("case_id") == CASE_ID]
    _require(len(rows) == 1, "directional E05 manifest row is not unique")
    case = rows[0]
    validate_case_row(case, args.repo_root.resolve())
    source = _git_identity(args.repo_root.resolve(), args.expected_commit)
    allocation = allocation_record()
    runtime = _runtime_imports(include_aegis=False)
    geometry_config = load_shadow_config(sources["geometry_config"])
    env = probe_env = None
    try:
        env, probe_env, task, observation, setup = _build_pair(runtime, case)
        pairing = pairing_record(
            case=case,
            selected_initial_state=setup["selected_initial_state"],
            settled_observation=observation,
            task_description=str(task.language),
            active_obstacle_name=setup["obstacle_name"],
            settled_simulator_state=np.asarray(env.sim.get_state().flatten(), dtype=np.float64),
        )
        _require(pairing == dataset["pairing"], "directional pairing differs")
        perception = archived["perception"]
        geometry = MultilinkEllipsoidShadow.from_aegis_geometry(
            geometry_config,
            {"p2": perception["mvee_center"], "R2": perception["mvee_rotation"], "Q2_diag": perception["mvee_semiaxes"], "record": {"label": perception["obstacle_label"]}},
        )
        probe = SubstepEightConstraintProbe(
            probe_env, geometry, active_obstacle_name=setup["obstacle_name"],
            quadratic_tolerance=1e-6, contact_distance_threshold_m=0.0,
            obstacle_primitive_union=None,
        )
        actions = successful["actions"]
        snapshots = {}
        for step in range(max(groups["test_steps"]) + 1):
            if step in groups["test_steps"]:
                snapshots[step] = _snapshot_env(env)
            env.step(_source_action(actions[step], step).tolist())
        radius = float(config["matched_random_test"]["radius_action"])
        random_count = int(config["matched_random_test"]["random_direction_count_per_state"])
        state_results = []
        learned_records = []
        random_records = []
        for step in groups["test_steps"]:
            _restore_env(env, snapshots[step])
            query = _successful_query_action(actions[step], step)
            nominal_rows = [item for item in dataset["records"] if item["state_step"] == step and item["candidate_source"] == "nominal"]
            _require(len(nominal_rows) == 1, "directional nominal record differs")
            nominal_transition = probe.transition(env, query)
            recomputed = feature_vector(nominal_transition["substeps"][0], query[:3], query[:3])
            stored = np.asarray(nominal_rows[0]["feature_vector"], dtype=np.float64)
            _require(float(np.max(np.abs(recomputed - stored))) <= 1e-10, "directional nominal feature differs")
            nominal_prediction = _predict(model, stored, artifact["feature_mean"], artifact["feature_standard_deviation"])
            gradient = nominal_prediction["gradient_xyz"]
            norm = float(np.linalg.norm(gradient))
            _require(norm > 1e-12, "directional gradient is zero")
            negative = -gradient / norm
            positive = gradient / norm

            def evaluate(direction, arm, index):
                xyz = query[:3] + radius * np.asarray(direction, dtype=np.float64)
                _require(np.all(np.abs(xyz) <= 1.0 + 1e-12), "directional candidate exceeds action bounds")
                action = query.copy()
                action[:3] = xyz
                transition = probe.transition(env, action)
                return {
                    "state_step": int(step), "arm": arm, "direction_index": index,
                    "radius_action": radius, "direction": np.asarray(direction).tolist(),
                    "candidate_xyz": xyz.tolist(), "contact_burden": contact_burden(transition),
                    "next_state_sha256": transition["next_state_sha256"],
                    "env_step_wall_seconds": float(transition["env_step_wall_seconds"]),
                }

            learned_negative = evaluate(negative, "revised_negative_gradient", None)
            learned_positive = evaluate(positive, "revised_positive_gradient", None)
            learned_records.extend([learned_negative, learned_positive])
            negative_feature = stored.copy(); negative_feature[-3:] = learned_negative["candidate_xyz"]
            positive_feature = stored.copy(); positive_feature[-3:] = learned_positive["candidate_xyz"]
            negative_risk = _predict(model, negative_feature, artifact["feature_mean"], artifact["feature_standard_deviation"])["risk"]
            positive_risk = _predict(model, positive_feature, artifact["feature_mean"], artifact["feature_standard_deviation"])["risk"]
            _require(negative_risk < positive_risk, "directional revised model sign ordering failed")
            directions = sample_feasible_unit_directions(
                query[:3], radius=1.0, count=random_count,
                seed=int(config["matched_random_test"]["seeds_by_state"][str(step)]),
                action_limit=1.0, maximum_attempts=100000,
            )
            prior_random = [item for item in random_result["random_records"] if item["state_step"] == step and item["radius_action"] == radius]
            _require(len(prior_random) == random_count, "directional prior random population differs")
            state_random = []
            for index, direction in enumerate(directions):
                old = next(item for item in prior_random if item["direction_index"] == index)
                current = evaluate(direction, "matched_random", index)
                _require(np.array_equal(np.asarray(current["candidate_xyz"]), np.asarray(old["candidate_xyz"])), "directional random action receipt differs")
                _require(current["contact_burden"]["contact_count"] == old["raw_protected_contact_count"], "directional random contact receipt differs")
                _require(current["next_state_sha256"] == old["next_state_sha256"], "directional random state receipt differs")
                state_random.append(current)
            random_records.extend(state_random)
            empirical_p = empirical_burden_p(
                learned_negative["contact_burden"],
                [item["contact_burden"] for item in state_random],
            )
            state_results.append({
                "state_step": int(step), "nominal_risk": nominal_prediction["risk"],
                "gradient_xyz": gradient.tolist(), "gradient_norm": norm,
                "negative_risk_at_radius": negative_risk,
                "positive_risk_at_radius": positive_risk,
                "negative_contact_burden": learned_negative["contact_burden"],
                "positive_contact_burden": learned_positive["contact_burden"],
                "random_at_least_as_good_count": sum(
                    burden_key(item["contact_burden"]) <= burden_key(learned_negative["contact_burden"])
                    for item in state_random
                ),
                "empirical_burden_p": empirical_p,
                "matched_random_gate_pass": empirical_p <= float(config["matched_random_test"]["empirical_p_maximum"]),
            })
        validation_accuracy = training["pair_metrics"]["validation"]["informative_pair_accuracy"]
        validation_pass = bool(validation_accuracy is not None and validation_accuracy >= float(config["decision_gate"]["minimum_validation_informative_pair_accuracy"]))
        matched_pass = bool(all(item["matched_random_gate_pass"] for item in state_results))
        mechanism_go = bool(validation_pass and matched_pass)
        result = {
            "schema_version": RESULT_SCHEMA, "status": "complete", "scientific_result": True,
            "claim_scope": config["claim_scope"], "source": source, "allocation": allocation,
            "config": config, "pairing": pairing,
            "immutable_sources": {name: {"path": str(path), "file_sha256": expected_files[name]} for name, path in sources.items()},
            "pair_records": pairs, "training": training,
            "model": {"path": str(args.model.resolve()), "file_sha256": _file_sha256(args.model.resolve())},
            "state_results": state_results, "learned_records": learned_records,
            "random_records": random_records,
            "decision": {
                "validation_directional_gate_pass": validation_pass,
                "revised_gradient_beats_matched_random_on_both_unseen_states": matched_pass,
                "paired_directional_supervision_mechanism_go": mechanism_go,
                "closed_loop_authorized": False,
                "stop_reason": None if mechanism_go else "revised_gradient_failed_validation_or_matched_random_gate",
            },
            "total_fresh_rollout_count": len(learned_records) + len(random_records),
            "total_clone_env_step_wall_seconds": float(sum(item["env_step_wall_seconds"] for item in learned_records + random_records)),
            "wall_seconds": (time.perf_counter_ns() - started) * 1e-9,
        }
        result["result_payload_sha256"] = _hash_without(result, "result_payload_sha256")
        _atomic_write(args.output.resolve(), result)
        print(json.dumps(result["decision"], sort_keys=True))
    finally:
        if probe_env is not None:
            probe_env.close()
        if env is not None:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

