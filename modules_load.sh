#!/bin/bash
module purge

# Charger le compilateur Intel le plus récent
module load intel-oneapi-compilers/2025.3.1/none-none

# Charger l'implémentation MPI correspondante (crucial pour le lien avec le compilateur)
module load intel-oneapi-mpi/2021.17.1/intel-oneapi-compilers-2025.3.1

# Charger Eigen3 (version 5.0.1 optimisée pour ce compilateur)
# Cela évitera à CMake de devoir le télécharger via FetchContent
module load eigen/5.0.1/intel-oneapi-compilers-2025.3.1

# Charger les outils de build (CMake et Gmake)
module load cmake/3.31.9/gcc-15.1.0
module load gmake/4.4.1/gcc-15.1.0
