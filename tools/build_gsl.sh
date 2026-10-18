#!/usr/bin/env bash
# GSL from source, static only, into PREFIX: what the platform wheels link
# drvec C against (publish.yml, cibuildwheel's before-all on Linux and macOS).
#
#   bash tools/build_gsl.sh PREFIX
#
# GSL 2.7.1 is the library the oracle RUNS on (its bin/drvec loads Ubuntu
# 24.04's libgsl.so.27, 2.7.1, though it was compiled against 2.8's headers),
# and the one tests.yml's C build on ubuntu-latest links (libgsl-dev), where
# the bytes have matched the oracle's. From source because manylinux_2_28's
# gsl-devel ships no static archive, and because a Homebrew bottle targets the
# runner's macOS, not the wheel's. The tarball is checked against its sha256.
set -euo pipefail
PREFIX=${1:?usage: build_gsl.sh PREFIX}
GSL_VERSION=2.7.1
GSL_SHA256=dcb0fbd43048832b757ff9942691a8dd70026d5da0ff85601e52687f6deeb34b

if [ -f "$PREFIX/lib/libgsl.a" ] && [ -f "$PREFIX/lib/libgslcblas.a" ]; then
    echo "build_gsl: $PREFIX already has GSL"; exit 0
fi
WORK=$(mktemp -d)
cd "$WORK"
T=gsl-$GSL_VERSION.tar.gz
for url in "https://ftp.gnu.org/gnu/gsl/$T" "https://mirrors.kernel.org/gnu/gsl/$T"; do
    if curl -fsSL --retry 3 -o "$T" "$url"; then break; fi
done
if command -v sha256sum >/dev/null 2>&1; then
    got=$(sha256sum "$T" | cut -d' ' -f1)
else
    got=$(shasum -a 256 "$T" | cut -d' ' -f1)
fi
if [ "$got" != "$GSL_SHA256" ]; then
    echo "build_gsl: $T has sha256 $got, expected $GSL_SHA256" >&2; exit 1
fi
tar xzf "$T"
cd gsl-$GSL_VERSION
NJ=$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
./configure --prefix="$PREFIX" --disable-shared --enable-static CFLAGS="-O2 -fPIC" >/dev/null
make -j"$NJ" >/dev/null
make install >/dev/null
ls -la "$PREFIX/lib/libgsl.a" "$PREFIX/lib/libgslcblas.a"
rm -rf "$WORK"
