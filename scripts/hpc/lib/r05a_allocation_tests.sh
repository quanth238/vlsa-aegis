#!/usr/bin/env bash

# Execute the exact dependency-backed R05A suites from one reviewed registry.
# Both the H100 canary and the CPU apparatus regression call this function; the
# expected counts are never inferred from the observed unittest output.

crfs_run_r05a_allocation_tests() {
  if [ "$#" -ne 5 ]; then
    echo "usage: crfs_run_r05a_allocation_tests REGISTRY PYTHON TEST_DIR LOG SUITE_LOG_DIR" >&2
    return 2
  fi
  local registry=$1
  local python=$2
  local test_dir=$3
  local output_log=$4
  local suite_log_dir=$5
  local registry_contract registered_suites registered_suite_count
  local executed_suite_count tab pattern expected suite_log observed

  test -f "$registry" || { echo "missing R05A allocation-test registry: $registry" >&2; return 2; }
  test -x "$python" || { echo "missing R05A allocation-test Python: $python" >&2; return 2; }
  test -d "$test_dir" || { echo "missing R05A allocation-test directory: $test_dir" >&2; return 2; }
  test -d "$suite_log_dir" || { echo "missing R05A suite-log directory: $suite_log_dir" >&2; return 2; }
  command -v jq >/dev/null 2>&1 || {
    echo "jq is required for the R05A test registry" >&2
    return 2
  }

  registry_contract='type == "object"
    and keys == ["schema_version", "suites"]
    and .schema_version == "1.0"
    and (.suites | type == "array" and length > 0)
    and all(.suites[];
      type == "object"
      and keys == ["expected_tests", "pattern"]
      and (.pattern | type == "string" and test("^test_[A-Za-z0-9_]+\\.py$"))
      and (.expected_tests | type == "number" and floor == . and . > 0))
    and ([.suites[].pattern] | length == (unique | length))'
  jq -e "$registry_contract" "$registry" >/dev/null || {
    echo "R05A allocation-test registry failed its strict contract" >&2
    return 2
  }
  if ! registered_suites=$(jq -r \
    '.suites[] | [.pattern, (.expected_tests | tostring)] | @tsv' "$registry"); then
    echo "cannot materialize the R05A allocation-test registry" >&2
    return 2
  fi
  if ! registered_suite_count=$(jq -r '.suites | length' "$registry"); then
    echo "cannot count the R05A allocation-test registry" >&2
    return 2
  fi

  : >"$output_log"
  executed_suite_count=0
  tab=$(printf '\t')
  while IFS="$tab" read -r pattern expected; do
    test -n "$pattern" && test -n "$expected" || {
      echo "R05A allocation-test registry emitted an empty row" >&2
      return 2
    }
    suite_log=$suite_log_dir/allocation-$pattern.log
    if ! "$python" -m unittest discover -s "$test_dir" -p "$pattern" -v \
      >"$suite_log" 2>&1; then
      echo "$pattern failed its dependency-backed allocation run" >&2
      return 2
    fi
    observed=$(sed -n 's/^Ran \([0-9][0-9]*\) tests\{0,1\} in .*/\1/p' "$suite_log")
    test "$observed" = "$expected" || {
      echo "$pattern ran ${observed:-unknown} tests; expected $expected" >&2
      return 2
    }
    test "$(grep -xc 'OK' "$suite_log")" = 1 || {
      echo "$pattern did not finish with exactly one OK" >&2
      return 2
    }
    if grep -Eq 'skipped=|^FAILED|^ERROR' "$suite_log"; then
      echo "$pattern skipped or failed allocation-backed tests" >&2
      return 2
    fi
    printf 'verified_test_suite=%s expected=%s observed=%s skips=0 status=passed\n' \
      "$pattern" "$expected" "$observed" >>"$output_log"
    sed "s/^/[$pattern] /" "$suite_log" >>"$output_log"
    executed_suite_count=$((executed_suite_count + 1))
  done <<EOF
$registered_suites
EOF
  test "$executed_suite_count" = "$registered_suite_count" || {
    echo "R05A allocation-test registry execution count changed" >&2
    return 2
  }
}
