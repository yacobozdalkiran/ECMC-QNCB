//
// Created by ozdalkiran-l on 1/28/26.
//

#include "heatbath_mpi.h"

#include "../observables/observables_mpi.h"
#include "heatbath.h"

//Performs a Heatbath hit on a link using the Cabbibo-Marinari algorithm
void mpi::heatbathcb::hit(GaugeField& field, const GeometryCB& geo, size_t site, int mu,
                          double beta, SU3& A, std::mt19937_64& rng) {
    field.compute_staple(geo, site, mu, A);

    SU3 W = field.view_link_const(site, mu) * A;
    SU2q wq = su2_to_quaternion(W.block<2, 2>(0, 0));
    SU2q r = ::heatbath::su2_step(beta, wq, rng);
    SU3 R = su2_quaternion_to_su3(r, 0, 1);
    field.view_link(site, mu) = R * field.view_link(site, mu);

    W = field.view_link_const(site, mu) * A;
    wq = su2_to_quaternion(W.block<2, 2>(1, 1));
    r = ::heatbath::su2_step(beta, wq, rng);
    R = su2_quaternion_to_su3(r, 1, 2);
    field.view_link(site, mu) = R * field.view_link(site, mu);

    W = field.view_link_const(site, mu) * A;
    SU2 wsu2;
    wsu2 << W(0, 0), W(0, 2), W(2, 0), W(2, 2);
    wq = su2_to_quaternion(wsu2);
    r = ::heatbath::su2_step(beta, wq, rng);
    R = su2_quaternion_to_su3(r, 0, 2);
    field.view_link(site, mu) = R * field.view_link(site, mu);
}

//Performs a Heatbath sweep on the non-frozen links
void mpi::heatbathcb::sweep(GaugeField& field, const GeometryCB& geo, double beta, int N_hits,
                            std::mt19937_64& rng) {
    SU3 A;
    for (int t = 1; t <= geo.L_int; t++) {
        for (int z = 1; z <= geo.L_int; z++) {
            for (int y = 1; y <= geo.L_int; y++) {
                for (int x = 1; x <= geo.L_int; x++) {
                    size_t site = geo.index(x, y, z, t);
                    for (int mu = 0; mu < 4; mu++) {
                        if (!geo.is_frozen(site, mu)) {
                            for (int h = 0; h < N_hits; h++) {
                                hit(field, geo, site, mu, beta, A, rng);
                            }
                        }
                    }
                }
            }
        }
    }
}

//Generates Heatbath samples according to input parameters
void mpi::heatbathcb::samples(GaugeField& field, const GeometryCB& geo,
                                             MpiTopology& topo, const HbParams& params,
                                             std::mt19937_64& rng) {
    std::vector<double> meas(params.N_samples);
    for (int m = 0; m < params.N_samples; m++) {
        // Update
            for (int s = 0; s < params.N_sweeps; s++) {
                sweep(field, geo, params.beta, params.N_hits, rng);
            }
    }
}
