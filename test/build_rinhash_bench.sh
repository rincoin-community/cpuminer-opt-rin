#!/bin/sh
# Builds test/rinhash_bench (links the miner's rinhash_core object) and
# test/rinhash_bench_prof (same code with per-stage counters). Run "make" first.
set -e
cd "$(dirname "$0")/.."
R=algo/rinhash
CFLAGS="${CFLAGS:--O3 -march=native}"
INC="-I$R -I$R/argon2/include -I$R/argon2 -I$R/argon2/blake2"
LIB="$R/argon2/cpuminer-core.o $R/argon2/cpuminer-argon2.o
     $R/argon2/cpuminer-encoding.o $R/argon2/cpuminer-ref.o
     $R/argon2/cpuminer-thread.o $R/argon2/cpuminer-argon2_dispatch.o
     $R/argon2/cpuminer-argon2_ssse3.o $R/argon2/cpuminer-argon2_avx2.o
     $R/argon2/cpuminer-argon2_avx512.o $R/argon2/blake2/cpuminer-blake2b.o
     $(ls $R/blake3/cpuminer-*.o) algo/verthash/tiny_sha3/cpuminer-sha3.o"
${CC:-gcc} $CFLAGS $INC test/rinhash_bench.c $R/cpuminer-rinhash_core.o $LIB \
   -no-pie -lpthread -o test/rinhash_bench
${CC:-gcc} $CFLAGS $INC -DRINHASH_PROFILE test/rinhash_bench.c \
   $R/rinhash_core.c $LIB -no-pie -lpthread -o test/rinhash_bench_prof
