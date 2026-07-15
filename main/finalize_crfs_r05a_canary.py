#!/usr/bin/env python3
"""Finalize measured memory into an atomic R05A canary results.json."""

from __future__ import annotations

import argparse

from crfs_oracle.r05a_canary import finalize_r05a_canary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--host-cgroup-peak-bytes", required=True, type=int)
    parser.add_argument("--host-cgroup-diagnostic", required=True)
    parser.add_argument("--gpu-samples", required=True)
    parser.add_argument("--allocation-tests-log", required=True)
    parser.add_argument("--allocation-tests-exit-code", required=True, type=int)
    args = parser.parse_args()
    output = finalize_r05a_canary(
        args.payload,
        args.output,
        host_cgroup_peak_bytes=args.host_cgroup_peak_bytes,
        host_cgroup_diagnostic_path=args.host_cgroup_diagnostic,
        gpu_samples_path=args.gpu_samples,
        allocation_tests_log=args.allocation_tests_log,
        allocation_tests_exit_code=args.allocation_tests_exit_code,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
