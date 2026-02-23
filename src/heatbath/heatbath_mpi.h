//
// Created by ozdalkiran-l on 1/28/26.
//

#ifndef ECMC_MPI_HEATBATH_MPI_H
#define ECMC_MPI_HEATBATH_MPI_H

#include "../gauge/GaugeField.h"
#include "../io/params.h"
#include "../mpi/MpiTopology.h"
#include "../su3/utils.h"

namespace mpi::heatbathcb {
void hit(GaugeField& field, const GeometryCB& geo, size_t site, int mu, double beta, SU3& A,
         std::mt19937_64& rng);
void sweep(GaugeField& field, const GeometryCB& geo, double beta, int N_hits, std::mt19937_64& rng);
void samples(GaugeField& field, const GeometryCB& geo, MpiTopology& topo,
                            const HbParams& params, std::mt19937_64& rng);
}  // namespace mpi::heatbathcb

#endif  // ECMC_MPI_HEATBATH_MPI_H
