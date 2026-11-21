#!/bin/sh
# Builds the RinHash verification and benchmark tool against the objects of a
# finished miner build.
#
#    test/build_rinhash_bench.sh [build directory]
#
# The build directory defaults to the source directory (a build in place).
# Output, in the build directory: rinhash_bench links the miner's own
# rinhash_core object, rinhash_bench_prof is the same code compiled with
# per-stage counters.
set -e
SRC=$(cd "$(dirname "$0")/.." && pwd)
B=$(cd "${1:-$SRC}" && pwd)
R=algo/rinhash

# Same compiler and flags as the miner build.
CC=${CC:-$(sed -n 's/^CC = //p' "$B/Makefile")}
CFLAGS=$(sed -n 's/^CFLAGS = //p' "$B/Makefile")
case "$(uname -m)" in
   x86_64|amd64) CFLAGS="$CFLAGS -mtune=generic" ;;
esac
case "$(uname -s)" in
   Linux) NOPIE=-no-pie ;;
   *)     NOPIE= ;;
esac

INC="-I$SRC/$R -I$SRC/$R/argon2/include -I$SRC/$R/argon2 -I$SRC/$R/argon2/blake2"
LIB="$(ls "$B"/$R/argon2/cpuminer-*.o "$B"/$R/argon2/blake2/cpuminer-*.o \
          "$B"/$R/blake3/cpuminer-*.o) $B/algo/verthash/tiny_sha3/cpuminer-sha3.o"

OUT=$B
[ "$B" = "$SRC" ] && OUT=$SRC/test

$CC $CFLAGS $INC "$SRC/test/rinhash_bench.c" "$B/$R/cpuminer-rinhash_core.o" \
   $LIB $NOPIE -lpthread -o "$OUT/rinhash_bench"
$CC $CFLAGS $INC -DRINHASH_PROFILE "$SRC/test/rinhash_bench.c" \
   "$SRC/$R/rinhash_core.c" $LIB $NOPIE -lpthread -o "$OUT/rinhash_bench_prof"
echo "built $OUT/rinhash_bench and $OUT/rinhash_bench_prof"
