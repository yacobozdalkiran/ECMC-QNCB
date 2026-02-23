//
// Created by ozdalkiran-l on 1/21/26.
//

#ifndef ECMC_MPI_HEATBATH_H
#define ECMC_MPI_HEATBATH_H

#include "../gauge/GaugeField.h"
#include "../su3/utils.h"

namespace heatbath {
    SU2q generate_su2_kp(double beta, double k, std::mt19937_64 &rng);
    SU2q su2_step(double beta, const SU2q &w, std::mt19937_64 &rng);
    void hit(GaugeField &field, const GeometryCB &geo, size_t site, int mu, double beta, SU3 &A, std::mt19937_64 &rng);
    void sweep(GaugeField &field, const GeometryCB &geo, double beta, int N_hits, std::mt19937_64 &rng);
}


#endif //ECMC_MPI_HEATBATH_H
