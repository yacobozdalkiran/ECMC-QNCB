//
// Created by ozdalkiran-l on 1/9/26.
//

#ifndef INC_4D_MPI_MPITOPOLOGY_H
#define INC_4D_MPI_MPITOPOLOGY_H

#include <mpi.h>

enum parity{
    even,
    odd
};

namespace mpi {
    class MpiTopology {
    public:
        int n_core_dim;
        int rank, size;
        int local_rank;
        int local_coords[4];
        parity p;
        int x0, xL, y0, yL, z0, zL, t0, tL;
        MPI_Comm cart_comm{};

        MpiTopology(int n_core_dim_);
    };
}

#endif //INC_4D_MPI_MPITOPOLOGY_H
