#!/bin/bash
# Checks a finished build, used by the release workflow.
#
#    test/ci-check.sh <cpuminer binary> <build directory> [more build directories]
#
# For every build directory the RinHash verification tool is built and run:
# it compares the optimized code with the public BLAKE3 and Argon2 APIs. The
# miner itself is then run in benchmark mode; a multi-level binary is run
# once per CPU level this machine supports.
set -euo pipefail
SRC=$(cd "$(dirname "$0")/.." && pwd)
MINER=$1
shift

# Extra miner options, for example a CPU affinity mask on a busy machine.
MINER_ARGS=${MINER_ARGS:-}
LOG=$(mktemp)
trap 'rm -f "$LOG"' EXIT

# Runs the miner in benchmark mode into $LOG, killing it if it hangs.
bench() {        # seconds algorithm
   local limit=$1 algo=$2 pid watchdog rc=0
   # shellcheck disable=SC2086
   "$MINER" -a "$algo" --benchmark -t 2 --time-limit "$limit" $MINER_ARGS > "$LOG" 2>&1 &
   pid=$!
   ( sleep 90; kill -9 $pid 2> /dev/null ) &
   watchdog=$!
   wait $pid || rc=$?
   kill $watchdog 2> /dev/null || true
   tr -d '\r' < "$LOG" | sed 's/\x1b\[[0-9;]*m//g' > "$LOG.txt"
   mv "$LOG.txt" "$LOG"
   return $rc
}

run_miner() {    # algorithm
   bench 3 "$1" || { tail -20 "$LOG"; echo "FAILED: $1 exited with an error or hung"; exit 1; }
   if grep -qi "self test" "$LOG"; then
      grep -i "self test" "$LOG"; echo "FAILED: RinHash self test"; exit 1
   fi
   grep -E "Multi-level|RinHash:|Benchmark:" "$LOG" | sed "s/^/   $1: /"
   grep -q "Benchmark:" "$LOG" || { tail -20 "$LOG"; echo "FAILED: $1 printed no benchmark result"; exit 1; }
}

for dir in "$@"; do
   echo "== RinHash verification, objects of $dir"
   "$SRC/test/build_rinhash_bench.sh" "$dir" > /dev/null
   out=$dir
   [ "$(cd "$dir" && pwd)" = "$SRC" ] && out=$SRC/test
   if "$out/rinhash_bench" 4000; then :; else
      rc=$?
      # A build for a CPU level this machine lacks cannot run here.
      if [ "${ALLOW_ILLEGAL:-0}" = 1 ] && [ $rc -ge 128 ]; then
         echo "   skipped: this CPU cannot run that level"
      else
         echo "FAILED: verification of $dir"; exit 1
      fi
   fi
done

bench 1 rinhash || true
max=$( sed -n 's/.*level [0-9] of the \([0-9]\) this CPU supports.*/\1/p' "$LOG" | head -1 )
if [ -n "$max" ]; then
   for level in $( seq 1 "$max" ); do
      echo "== miner at CPU level $level of $max"
      export CPUMINER_LEVEL=$level
      for algo in rinhash sha256d x11 keccak argon2d500; do run_miner $algo; done
   done
   unset CPUMINER_LEVEL
else
   echo "== miner"
   for algo in rinhash sha256d x11 keccak argon2d500; do run_miner $algo; done
fi
echo "all checks passed"
