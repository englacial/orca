#!/usr/bin/env bash
# Build and install the HIL fixture's UHD from a local source checkout.
# This is the only supported way to (re)build UHD on the fixture; the HIL
# harness never calls it. Run it from inside the activated conda env so UHD
# and sdr/ share one Boost and one compiler.
#
# Usage: build_uhd.sh [source_dir] [install_prefix]
#   defaults: ~/uhd  ~/uhd-patched
set -euo pipefail

SRC=$(realpath "${1:-$HOME/uhd}")
PREFIX=$(realpath -m "${2:-$HOME/uhd-patched}")
BUILD="$SRC/host/build"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "error: activate the ORCA conda env first" >&2
    exit 1
fi
if compgen -G "$CONDA_PREFIX/lib/libuhd.so*" > /dev/null; then
    echo "error: $CONDA_PREFIX has its own libuhd; remove it (conda remove --force uhd)" >&2
    exit 1
fi

# Flags for conda's protobuf/gRPC CMake configs and libusb location
cmake -S "$SRC/host" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_INSTALL_RPATH="$PREFIX/lib;$CONDA_PREFIX/lib" \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DCMAKE_FIND_PACKAGE_PREFER_CONFIG=ON \
    -DProtobuf_PROTOC_EXECUTABLE="$CONDA_PREFIX/bin/protoc" \
    -DGRPC_CPP_PLUGIN="$CONDA_PREFIX/bin/grpc_cpp_plugin" \
    -DLIBUSB_INCLUDE_DIRS="$CONDA_PREFIX/include/libusb-1.0" \
    -DLIBUSB_LIBRARIES="$CONDA_PREFIX/lib/libusb-1.0.so" \
    -DENABLE_B200=ON -DENABLE_USB=ON -DENABLE_UTILS=ON \
    -DENABLE_PYTHON_API=OFF -DENABLE_TESTS=OFF -DENABLE_EXAMPLES=OFF \
    -DENABLE_MANUAL=OFF -DENABLE_DOXYGEN=OFF

cmake --build "$BUILD" -j4
cmake --install "$BUILD"

UHD_IMAGES_DIR="$PREFIX/share/uhd/images" \
    "$PREFIX/bin/uhd_images_downloader" -t b2xx -i "$PREFIX/share/uhd/images"

version=$("$PREFIX/bin/uhd_config_info" --version | head -1)

# Provenance stamp, checked by the HIL preflight test (T0)
commit=$(git -C "$SRC" rev-parse HEAD)
dirty=false
[[ -n "$(git -C "$SRC" status --porcelain)" ]] && dirty=true
compiler=$(grep -m1 '^CMAKE_CXX_COMPILER:' "$BUILD/CMakeCache.txt" | cut -d= -f2-)
boost=$(grep -m1 '#define BOOST_LIB_VERSION' "$CONDA_PREFIX/include/boost/version.hpp" | awk '{print $3}' | tr -d '"')
cat > "$PREFIX/ORCA_UHD_BUILD.yaml" <<EOF
source_dir: $SRC
commit: $commit
dirty: $dirty
built_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)
uhd_version: "$version"
compiler: "$compiler $("$compiler" -dumpfullversion)"
boost_version: "$boost"
conda_prefix: $CONDA_PREFIX
EOF
cat "$PREFIX/ORCA_UHD_BUILD.yaml"
