#!/usr/bin/env bash
# setup_replica_renderer.sh
# Clones, patches, and builds ReplicaPoseRenderer from scratch.
# Run this once on a new machine to set up the rendering pipeline.
#
# Usage:
#   bash tools/setup_replica_renderer.sh
#
# After successful build, the binary will be at:
#   ~/tools/Replica-Dataset/build/ReplicaSDK/ReplicaPoseRenderer

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPLICA_SDK_DIR="${HOME}/tools/Replica-Dataset"

echo "=== Setting up ReplicaPoseRenderer ==="
echo "Target directory: ${REPLICA_SDK_DIR}"
echo ""

# ── 1. Clone ────────────────────────────────────────────────────────────────
if [ ! -d "${REPLICA_SDK_DIR}" ]; then
    echo "[1/7] Cloning Replica-Dataset..."
    git clone https://github.com/facebookresearch/Replica-Dataset.git "${REPLICA_SDK_DIR}"
else
    echo "[1/7] Replica-Dataset already cloned, skipping."
fi
cd "${REPLICA_SDK_DIR}"

# ── 2. Submodules ───────────────────────────────────────────────────────────
echo "[2/7] Initialising submodules (Pangolin, Eigen)..."
git submodule update --init --force

# ── 3. Bug fix: EGL.cpp — foundCudaDev never set to true ────────────────────
echo "[3/7] Applying bug fix to EGL.cpp..."
EGL_FILE="ReplicaSDK/ptex/EGL.cpp"
if ! grep -q "foundCudaDev = true;" "${EGL_FILE}"; then
    # Insert 'foundCudaDev = true;' before the 'break;' inside the cudaDevice match block
    sed -i 's/if (cudaDevNumber == cudaDevice) {/if (cudaDevNumber == cudaDevice) {\n        foundCudaDev = true;   \/\/ fix: was missing, caused X11 fallback on headless servers/' "${EGL_FILE}"
    echo "    EGL.cpp patched."
else
    echo "    EGL.cpp already patched, skipping."
fi

# ── 4. Bug fix: render.cpp — exposure/gamma for HDR textures ────────────────
echo "[4/7] Applying exposure/gamma fix to render.cpp..."
RENDER_FILE="ReplicaSDK/src/render.cpp"
if ! grep -q "SetExposure" "${RENDER_FILE}"; then
    sed -i 's/PTexMesh ptexMesh(meshFile, atlasFolder);/PTexMesh ptexMesh(meshFile, atlasFolder);\n\n  \/\/ HDR tone-mapping (default 1.0\/1.0 produces nearly-black images)\n  ptexMesh.SetExposure(0.025f);\n  ptexMesh.SetGamma(1.6969f);/' "${RENDER_FILE}"
    echo "    render.cpp patched."
else
    echo "    render.cpp already patched, skipping."
fi

# ── 5. Copy ReplicaPoseRenderer source and update CMakeLists ────────────────
echo "[5/7] Installing ReplicaPoseRenderer source..."
cp "${SCRIPT_DIR}/ReplicaSDK/src/replica_pose_renderer.cpp" ReplicaSDK/src/
if ! grep -q "ReplicaPoseRenderer" ReplicaSDK/CMakeLists.txt; then
    cat "${SCRIPT_DIR}/ReplicaSDK/CMakeLists_addition.txt" >> ReplicaSDK/CMakeLists.txt
    echo "    CMakeLists.txt updated."
else
    echo "    CMakeLists.txt already has ReplicaPoseRenderer target, skipping."
fi

# ── 6. System dependencies ──────────────────────────────────────────────────
echo "[6/7] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y libgl1-mesa-dev libglew-dev libglfw3-dev libegl1-mesa-dev

# ── 7. Build ────────────────────────────────────────────────────────────────
echo "[7/7] Building Pangolin + ReplicaSDK..."

# Pangolin
cd 3rdparty/Pangolin
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PANGOLIN_PYTHON=OFF -DCMAKE_BUILD_PARALLEL_LEVEL=4
make -j4
cd "${REPLICA_SDK_DIR}"

# ReplicaSDK
mkdir -p build && cd build
cmake ..
make -j4 ReplicaPoseRenderer ReplicaRenderer

echo ""
echo "=== Build complete ==="
echo "Binaries:"
echo "  ${REPLICA_SDK_DIR}/build/ReplicaSDK/ReplicaRenderer"
echo "  ${REPLICA_SDK_DIR}/build/ReplicaSDK/ReplicaPoseRenderer"
echo ""
echo "Required env var for headless NVIDIA rendering:"
echo "  export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
