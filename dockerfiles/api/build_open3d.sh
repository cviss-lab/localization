#!/bin/bash

set -ex

if [ ! $(grep open3d requirements.txt) ]; then
  echo "open3d is not found/required"
  exit 0
fi

version="v$(grep -oP 'open3d==\K.*' requirements.txt)"
open3d_dir="$(mktemp --directory)"

git clone -b "$version" https://github.com/isl-org/Open3D.git "$open3d_dir"
pushd "$open3d_dir"

SUDO=" " ./util/install_deps_ubuntu.sh assume-yes

# http://www.open3d.org/docs/latest/tutorial/visualization/headless_rendering.html
pip install numpy matplotlib

mkdir build && cd build

cmake \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_ISPC_MODULE=OFF \
  -DBUILD_GUI=OFF \
  -DBUILD_WEBRTC=OFF \
  -DENABLE_HEADLESS_RENDERING=ON \
  -DENABLE_CACHED_CUDA_MANAGER=OFF \
  -DUSE_SYSTEM_GLEW=OFF \
 ..

make -j$(nproc)
make install-pip-package

popd
rm -rf "$open3d_dir"
