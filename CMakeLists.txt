cmake_minimum_required(VERSION 3.10)
project(ECMC_NQCB)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

find_package(MPI REQUIRED)
find_package(Eigen3 REQUIRED) # Décommente si besoin
include(FetchContent)
#Lime
FetchContent_Declare(
        lime
        GIT_REPOSITORY https://github.com/usqcd-software/c-lime.git
        GIT_TAG        master  # Ou un tag spécifique comme v1.3.2
)
FetchContent_MakeAvailable(lime)
#Eigen
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found. Download via FetchContent...")

    FetchContent_Declare(
            eigen
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG 3.4.0
    )
    FetchContent_MakeAvailable(eigen)
else()
    message(STATUS "Eigen3 found on the system.")
endif()

#===ECMC QCB===
add_executable(gauge_ecmc_ncb src/bin/gauge_ecmc_ncb.cpp
        src/gauge/GaugeField.cpp
        src/su3/utils.cpp
        src/geometry/GeometryCB.cpp
        src/geometry/types.h
        src/mpi/MpiTopology.cpp
        src/mpi/HalosExchange.cpp
        src/observables/observables_mpi.cpp
        src/mpi/Shift.cpp
        src/ecmc/ecmc_mpi_cb.cpp
        src/io/params.h
        src/io/io.cpp
        src/flow/gradient_flow.cpp
        src/io/ildg.cpp
)
target_link_libraries(gauge_ecmc_ncb PRIVATE MPI::MPI_CXX Eigen3::Eigen lime)
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    target_compile_options(gauge_ecmc_ncb PRIVATE
            -O3
            -xHost            # Remplace -march=native pour Intel
            -qopenmp
            -fp-model precise
            -Wall -Wextra
    )
else()
    target_compile_options(gauge_ecmc_ncb PRIVATE
            -O2
            #-ffast-math
            #-march=native
            -Wall -Wextra -Wpedantic
    )
endif()

#===HB QCB===
add_executable(gauge_hb_ncb src/bin/gauge_hb_ncb.cpp
        src/gauge/GaugeField.cpp
        src/su3/utils.cpp
        src/geometry/GeometryCB.cpp
        src/geometry/types.h
        src/mpi/MpiTopology.cpp
        src/mpi/HalosExchange.cpp
        src/observables/observables_mpi.cpp
        src/mpi/Shift.cpp
        src/heatbath/heatbath.cpp
        src/heatbath/heatbath_mpi.cpp
        src/io/params.h
        src/io/io.cpp
        src/flow/gradient_flow.cpp
        src/io/ildg.cpp
)
target_link_libraries(gauge_hb_ncb PRIVATE MPI::MPI_CXX Eigen3::Eigen lime)
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    target_compile_options(gauge_hb_ncb PRIVATE
            -O3
            -xHost            # Remplace -march=native pour Intel
            -qopenmp
            -fp-model precise
            -Wall -Wextra
    )
else()
    target_compile_options(gauge_hb_ncb PRIVATE
            -O2
            #-ffast-math
            #-march=native
            -Wall -Wextra -Wpedantic
    )
endif()

