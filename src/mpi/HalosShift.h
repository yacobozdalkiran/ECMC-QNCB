//
// Created by ozdalkiran-l on 1/12/26.
//

#ifndef INC_4D_MPI_HALO_H
#define INC_4D_MPI_HALO_H

#include "../gauge/GaugeField.h"

//Halos used to shift the gauge configurations between all nodes
class HalosShift {
public:
    std::vector<Complex> send;
    std::vector<Complex> recv;
    int L_int; //Size of the internal square lattice
    int V_int;

    explicit HalosShift(const GeometryCB &geo) {
        L_int = geo.L_int;
        V_int = geo.V_int;
        send.resize(V_int*4*9);
        recv.resize(V_int*4*9);
    }
};

#endif
