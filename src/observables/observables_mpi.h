#ifndef INC_4D_MPI_OBSERVABLES_H
#define INC_4D_MPI_OBSERVABLES_H

#include "../gauge/GaugeField.h"
#include "../mpi/MpiTopology.h"
#include "../flow/gradient_flow.h"

namespace mpi::observables {
double mean_plaquette_local(const GaugeField& field, const GeometryCB& geo);
double mean_plaquette_global(GaugeField& field, const GeometryCB& geo, MpiTopology& topo);
SU3 clover_site(const GaugeField& field, const GeometryCB& geo, size_t site, int mu, int nu);
int levi_civita(int mu, int nu, int rho, int sigma);
std::pair<double, double> local_q_e_clover(const GaugeField& field, const GeometryCB& geo,
                                           size_t site);
std::pair<double, double> topo_q_e_clover(const GaugeField& field, const GeometryCB& geo);
std::pair<double, double> topo_q_e_clover_global(const GaugeField& field, const GeometryCB& geo,
                                                 MpiTopology& topo);
std::vector<double> topo_charge_flowed(GaugeField& field, const GeometryCB& geo, GradientFlow& gf, mpi::MpiTopology& topo, int N_steps_gf, int N_rk_steps);
}  // namespace mpi::observables

#endif  // INC_4D_MPI_OBSERVABLES_H
