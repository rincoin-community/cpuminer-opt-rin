#!/bin/bash
# Packs one build for the release workflow.
#
#    test/ci-package.sh <platform> <cpuminer binary>     writes dist/<archive>
#    test/ci-package.sh --tag                            prints the release tag
#
# The tag is v<version in configure.ac>-rin.<short commit>.
set -euo pipefail
SRC=$(cd "$(dirname "$0")/.." && pwd)
version=$(sed -n 's/^AC_INIT(\[cpuminer-opt\], *\[\(.*\)\]).*/\1/p' "$SRC/configure.ac")
# GITHUB_SHA is set in the workflow; the MSYS2 shell there has no git.
commit=${GITHUB_SHA:-$(git -C "$SRC" rev-parse HEAD)}
commit=$(echo "$commit" | cut -c1-7)
tag="v$version-rin.$commit"

if [ "$1" = "--tag" ]; then
   echo "$tag"
   exit 0
fi

platform=$1
binary=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
name=cpuminer-opt-rin-$tag-$platform
rm -rf "$SRC/dist"
mkdir -p "$SRC/dist/$name"
cp "$binary" "$SRC/dist/$name/"
cp "$SRC/README.md" "$SRC/RELEASE_NOTES" "$SRC/COPYING" "$SRC/LICENSE" "$SRC/dist/$name/"

cd "$SRC/dist"
case $platform in
   windows-*)
      # The DLLs of the MinGW runtime and of libcurl that the binary loads.
      ldd "$binary" | awk '{ print $3 }' | grep -i '/ucrt64/\|/mingw64/' | sort -u \
         | while read -r dll; do cp "$dll" "$name/"; done
      zip -q -r "$name.zip" "$name"
      ;;
   *)
      tar -czf "$name.tar.gz" "$name"
      ;;
esac
rm -rf "$name"
ls -l
