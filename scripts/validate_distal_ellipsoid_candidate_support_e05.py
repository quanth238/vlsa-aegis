#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path
from scripts.evaluate_distal_ellipsoid_candidate_support_e05 import RESULT_SCHEMA, _hash_without, _load_config
from scripts.replay_distal_three_ellipsoid_multicbf import _atomic_write, _load, _require


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--result",type=Path,required=True); parser.add_argument("--config",type=Path,required=True); parser.add_argument("--expected-commit",required=True); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
    config=_load_config(args.config.resolve()); result=_load(args.result.resolve())
    _require(result.get("schema_version")==RESULT_SCHEMA and result.get("config")==config,"result contract differs")
    _require(result.get("source",{}).get("commit")==args.expected_commit and result.get("source",{}).get("dirty") is False,"source differs")
    _require(result.get("result_payload_sha256")==_hash_without(result,"result_payload_sha256"),"result payload differs")
    states=result.get("state_results",[]); _require(len(states)==9 and {x["state_step"] for x in states}==set(range(184,193)),"state set differs")
    test=[x for x in states if x["state_step"] in config["test_steps"]]; support=bool(len(test)==2 and all(x["ellipsoid_safe_candidate_count"]>0 for x in test))
    _require(result["decision"]["ellipsoid_candidate_support_pass"] is support and result["decision"]["ellipsoid_margin_mlp_training_authorized"] is support,"decision differs")
    validation={"schema_version":"vlsa_distal_ellipsoid_candidate_support_e05_validation.v1","status":"passed","scientific_result":True,"expected_commit":args.expected_commit,"result_payload_sha256":result["result_payload_sha256"],"ellipsoid_candidate_support_pass":support,"ellipsoid_margin_mlp_training_authorized":support}
    validation["validation_payload_sha256"]=hashlib.sha256(json.dumps(validation,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest(); _atomic_write(args.output.resolve(),validation); print(json.dumps(validation,sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())
