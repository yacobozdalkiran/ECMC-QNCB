#!/bin/bash
source modules_load.sh
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_C_COMPILER=mpiicx -DCMAKE_CXX_COMPILER=mpiicpx -DCMAKE_BUILD_TYPE=Release
make -j $(nproc)
cd ..
