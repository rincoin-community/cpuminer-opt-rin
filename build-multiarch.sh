#!/bin/bash
#
# Builds one cpuminer for x86-64 (Linux or Windows/MSYS2) that contains the
# whole miner compiled for four CPU levels and picks one when it starts.
#
#   level 1  -march=x86-64            SSE2
#   level 2  -march=x86-64-v2         SSSE3, SSE4.2
#   level 3  -march=x86-64-v3 -maes   AVX2
#   level 4  -march=x86-64-v4 -maes   AVX-512
#
# How: the source is configured and built four times out of tree, once per
# level. The objects of the algorithm code of each build get their global
# symbols prefixed with ml<level>_ and go into one archive per level. The
# miner framework is taken from the level 1 build and linked with the four
# archives and multiarch.c, which forwards the few calls from the framework
# into the algorithm code to the level the CPU supports.
#
# Needs gcc 12 or newer (the x86-64-v2/v3/v4 names), binutils and the usual
# cpuminer build dependencies. The source directory must not be configured
# in place (run "make distclean" there first).
#
# Environment: OUT (build directory, default ./build-multiarch), JOBS,
# CC, CXX, CONFIGURE_ARGS (default --with-curl), BASE_CFLAGS (default -O3 -Wall).

set -euo pipefail
export LC_ALL=C

SRC=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$SRC/build-multiarch}
JOBS=${JOBS:-$(nproc)}
CC=${CC:-gcc}
CXX=${CXX:-g++}
CONFIGURE_ARGS=${CONFIGURE_ARGS:---with-curl}
BASE_CFLAGS=${BASE_CFLAGS:--O3 -Wall}

# MinGW reaches variables of other objects through per-object pointer stubs
# (.refptr.<name>) unless told that everything is within 2 GiB. The stubs
# are merged by section name, which would join the four copies of the
# algorithm code. The small code model does without them.
case "$("$CC" -dumpmachine)" in
   *mingw*) BASE_CFLAGS="$BASE_CFLAGS -mcmodel=small" ;;
esac

LEVEL_FLAGS=( "" "-march=x86-64" "-march=x86-64-v2" "-march=x86-64-v3 -maes"
              "-march=x86-64-v4 -maes" )

# Objects of the framework: compiled once, everything else is algorithm code.
FRAMEWORK='cpuminer-(cpu-miner|util|api|sysinfos|malloc-huge|dummy|winansi)\.o(bj)?$'

if [ -f "$SRC/config.status" ]; then
   echo "The source directory is configured in place: run 'make distclean' first." >&2
   exit 1
fi
if [ ! -f "$SRC/Makefile.in" ]; then
   ( cd "$SRC" && ./autogen.sh )
fi

mkdir -p "$OUT"
for L in 1 2 3 4; do
   B=$OUT/l$L
   mkdir -p "$B"
   if [ ! -f "$B/Makefile" ]; then
      ( cd "$B" && "$SRC/configure" CC="$CC" CXX="$CXX" \
           CFLAGS="$BASE_CFLAGS ${LEVEL_FLAGS[$L]}" $CONFIGURE_ARGS )
   fi
   make -C "$B" -j "$JOBS"

   # Prefix every global symbol the algorithm objects define.
   ( cd "$B" && find . -name 'cpuminer-*.o' | grep -Ev "$FRAMEWORK" | sort ) \
      > "$OUT/l$L.objects"
   ( cd "$B" && xargs nm -g --defined-only --format=posix < "$OUT/l$L.objects" ) \
      | awk -v p="ml${L}_" 'NF >= 2 && $1 !~ /:$/ { print $1, p $1 }' \
      | sort -u > "$OUT/l$L.symbols"
   if grep -q '^\.refptr\.' "$OUT/l$L.symbols"; then
      echo "level $L: the objects contain .refptr stubs, see the note on MinGW above" >&2
      exit 1
   fi
   rm -rf "$OUT/l$L.renamed" "$OUT/libalgo_l$L.a"
   mkdir -p "$OUT/l$L.renamed"
   while read -r obj; do
      objcopy --redefine-syms="$OUT/l$L.symbols" "$B/$obj" \
              "$OUT/l$L.renamed/$(echo "${obj#./}" | tr '/' '_')"
   done < "$OUT/l$L.objects"
   ar rcs "$OUT/libalgo_l$L.a" "$OUT"/l$L.renamed/*.o
   echo "level $L: $(wc -l < "$OUT/l$L.objects") objects," \
        "$(wc -l < "$OUT/l$L.symbols") symbols"
done

# Framework objects and link settings of the level 1 build.
B=$OUT/l1
FW=$( cd "$B" && find . -name 'cpuminer-*.o' | grep -E "$FRAMEWORK" \
      | sed "s|^\./|$B/|" | tr '\n' ' ' )
LINK=$( printf 'multiarch-link:\n\t@echo $(cpuminer_LDFLAGS) $(LDFLAGS) $(cpuminer_LDADD) $(LIBS)\n' \
        | make -s --no-print-directory -C "$B" -f Makefile -f - multiarch-link )
EXE=$( printf 'multiarch-exe:\n\t@echo cpuminer$(EXEEXT)\n' \
       | make -s --no-print-directory -C "$B" -f Makefile -f - multiarch-exe )

# The framework may call into the algorithm code only through the functions
# multiarch.c forwards.
FORWARDED="exec_hash_function get_algo_alias register_algo_gate sha256d sph_sha1_full sph_sha256_full"
# shellcheck disable=SC2086
NEEDED=$( nm -u $FW | awk 'NF == 2 { print $2 }' | sort -u \
          | comm -12 - <( cut -d' ' -f1 "$OUT/l1.symbols" | sort -u ) | tr '\n' ' ' )
if [ "$( echo $NEEDED )" != "$FORWARDED" ]; then
   echo "The framework references algorithm symbols multiarch.c does not forward:" >&2
   echo "  needed:    $NEEDED" >&2
   echo "  forwarded: $FORWARDED" >&2
   exit 1
fi

"$CC" $BASE_CFLAGS -march=x86-64 -c "$SRC/multiarch.c" -o "$OUT/multiarch.o"
# shellcheck disable=SC2086
( cd "$B" && "$CXX" -o "$OUT/$EXE" $FW "$OUT/multiarch.o" \
     -Wl,--start-group "$OUT"/libalgo_l4.a "$OUT"/libalgo_l3.a \
     "$OUT"/libalgo_l2.a "$OUT"/libalgo_l1.a -Wl,--end-group $LINK )

echo "built $OUT/$EXE"
