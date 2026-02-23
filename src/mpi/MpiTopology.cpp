//
// Created by ozdalkiran-l on 1/9/26.
//

#include "MpiTopology.h"

mpi::MpiTopology::MpiTopology(int n_core_dim_): local_coords{0,0,0,0} {
    rank = size = local_rank = x0 = xL = y0 = yL = z0 = zL = t0 = tL = 0;
    n_core_dim = n_core_dim_;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    //Creating cartesian topology
    int dims[4] = {n_core_dim, n_core_dim, n_core_dim, n_core_dim};
    int period[4] = {1, 1, 1, 1};
    int reorder = 1;
    MPI_Cart_create(MPI_COMM_WORLD, 4, dims, period, reorder, &cart_comm);
    MPI_Comm_rank(cart_comm, &local_rank);
    MPI_Cart_coords(cart_comm, local_rank, 4, local_coords);
    p = ((local_coords[0] + local_coords[1] + local_coords[2] + local_coords[3])%2==0)? even : odd;
    MPI_Cart_shift(cart_comm, 0, 1, &x0, &xL);
    MPI_Cart_shift(cart_comm, 1, 1, &y0, &yL);
    MPI_Cart_shift(cart_comm, 2, 1, &z0, &zL);
    MPI_Cart_shift(cart_comm, 3, 1, &t0, &tL);
}
