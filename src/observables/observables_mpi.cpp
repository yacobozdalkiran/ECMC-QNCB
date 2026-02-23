#include "observables_mpi.h"

#include "../io/io.h"
#include "../mpi/HalosExchange.h"
#include <iostream>

// Computation of mean plaquette with halos embedded in field (needs field halos exchange first)
double mpi::observables::mean_plaquette_local(const GaugeField& field, const GeometryCB& geo) {
    double sum = 0.0;
    SU3 U1, U2, U3, U4;
    for (int t = 1; t <= geo.L_int; t++) {
        for (int z = 1; z <= geo.L_int; z++) {
            for (int y = 1; y <= geo.L_int; y++) {
                for (int x = 1; x <= geo.L_int; x++) {
                    size_t site = geo.index(x, y, z, t);
                    for (int mu = 0; mu < 4; mu++) {
                        for (int nu = mu + 1; nu < 4; nu++) {
                            U1 = field.view_link_const(site, mu);
                            U2 = field.view_link_const(geo.get_neigh(site, mu, up), nu);
                            U3 = field.view_link_const(geo.get_neigh(site, nu, up), mu).adjoint();
                            U4 = field.view_link_const(site, nu).adjoint();
                            sum += (U1 * U2 * U3 * U4).trace().real() / 3.0;
                        }
                    }
                }
            }
        }
    }
    return sum;
}

// Computation of global mean plaquette with halos embedded in field (needs field halos exchanges
// first)
double mpi::observables::mean_plaquette_global(GaugeField& field, const GeometryCB& geo,
                                               MpiTopology& topo) {
    //mpi::exchange::exchange_halos_cascade(field, geo, topo);
    double local_mean_plaquette = mean_plaquette_local(field, geo);
    double global_mean_plaquette = 0.0;
    MPI_Allreduce(&local_mean_plaquette, &global_mean_plaquette, 1, MPI_DOUBLE, MPI_SUM,
                  topo.cart_comm);
    global_mean_plaquette /= 6.0 * geo.V_int * topo.size;
    return global_mean_plaquette;
}

//Computation of levi-civita tensor
int mpi::observables::levi_civita(int mu, int nu, int rho, int sigma) {
    // A stocker dans un tableau
    if (mu == nu || mu == rho || mu == sigma || nu == rho || nu == sigma || rho == sigma) return 0;
    int inv = 0;
    inv += (mu > nu);
    inv += (mu > rho);
    inv += (mu > sigma);
    inv += (nu > rho);
    inv += (nu > sigma);
    inv += (rho > sigma);

    return (inv & 1) ? -1 : +1;  // Parity test on inv
}

// Computes G_{\mu\nu}(site) clover
SU3 mpi::observables::clover_site(const GaugeField& field, const GeometryCB& geo, size_t site,
                                  int mu, int nu) {
    if (mu == nu) std::cerr << "mu = nu => G(site, mu, nu) = 0\n";
    size_t x = site;
    size_t xpmu = geo.get_neigh(x, mu, up);          // x+mu
    size_t xpnu = geo.get_neigh(x, nu, up);          // x+nu
    size_t xmmu = geo.get_neigh(x, mu, down);        // x-mu
    size_t xmnu = geo.get_neigh(x, nu, down);        // x-nu
    size_t xmmupnu = geo.get_neigh(xmmu, nu, up);    // x-mu+nu
    size_t xmmumnu = geo.get_neigh(xmmu, nu, down);  // x-mu-nu
    size_t xpmumnu = geo.get_neigh(xpmu, nu, down);  // x+mu-nu
    SU3 clover = SU3::Zero();
    clover += field.view_link_const(x, mu) * field.view_link_const(xpmu, nu) *
              field.view_link_const(xpnu, mu).adjoint() * field.view_link_const(x, nu).adjoint();
    clover += field.view_link_const(x, nu) * field.view_link_const(xmmupnu, mu).adjoint() *
              field.view_link_const(xmmu, nu).adjoint() * field.view_link_const(xmmu, mu);
    clover += field.view_link_const(xmmu, mu).adjoint() *
              field.view_link_const(xmmumnu, nu).adjoint() * field.view_link_const(xmmumnu, mu) *
              field.view_link_const(xmnu, nu);
    clover += field.view_link_const(xmnu, nu).adjoint() * field.view_link_const(xmnu, mu) *
              field.view_link_const(xpmumnu, nu) * field.view_link_const(x, mu).adjoint();
    SU3 F = (clover - clover.adjoint()).eval();
    F *= 0.125;
    return F;
}

// Computes the local clover topological charge and energy at site
std::pair<double, double> mpi::observables::local_q_e_clover(const GaugeField& field,
                                                             const GeometryCB& geo, size_t site) {
    SU3 F01 = clover_site(field, geo, site, 0, 1);
    SU3 F02 = clover_site(field, geo, site, 0, 2);
    SU3 F03 = clover_site(field, geo, site, 0, 3);
    SU3 F12 = clover_site(field, geo, site, 1, 2);
    SU3 F13 = clover_site(field, geo, site, 1, 3);
    SU3 F23 = clover_site(field, geo, site, 2, 3);

    // 2. On utilise la forme explicite de epsilon_{mu nu rho sigma}
    // Q ~ Tr(F01*F23 - F02*F13 + F03*F12)
    double q = (F01 * F23).trace().real() - (F02 * F13).trace().real() + (F03 * F12).trace().real();

    // On calcule aussi l'énergie locale
    double e_local = 0.5 * (F01.squaredNorm() + F02.squaredNorm() + F03.squaredNorm() +
                            F12.squaredNorm() + F13.squaredNorm() + F23.squaredNorm());
    // 3. Le facteur global est 1/(4*pi^2) car le 1/32 a été absorbé
    // par les combinaisons et le facteur 1/8 de F.
    return {q * (1.0 / (4.0 * M_PI * M_PI)), e_local};
}

// Return the clover topological charge and energy density
std::pair<double, double> mpi::observables::topo_q_e_clover(const GaugeField& field,
                                                            const GeometryCB& geo) {
    double q = 0.0;
    double e = 0.0;
    for (int t = 1; t <= geo.L_int; t++) {
        for (int z = 1; z <= geo.L_int; z++) {
            for (int y = 1; y <= geo.L_int; y++) {
                for (int x = 1; x <= geo.L_int; x++) {
                    size_t site = geo.index(x, y, z, t);
                    auto local = local_q_e_clover(field, geo, site);
                    q += local.first;
                    e += local.second;
                }
            }
        }
    }
    return {q, e};
}

//Returns topological charge and energy density of the whole lattice
std::pair<double, double> mpi::observables::topo_q_e_clover_global(const GaugeField& field,
                                                                   const GeometryCB& geo,
                                                                   MpiTopology& topo) {
    auto local_res = topo_q_e_clover(field, geo);
    double local_q = local_res.first;
    double local_e_total = local_res.second;

    double global_send_buffer[2] = {local_q, local_e_total};
    double global_recv_buffer[2] = {0.0, 0.0};

    // On somme les charges et les énergies de tous les rangs
    MPI_Allreduce(global_send_buffer, global_recv_buffer, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double total_q = global_recv_buffer[0];
    double total_e = global_recv_buffer[1];

    double total_volume = static_cast<double>(geo.V_int) * topo.size;

    // On retourne {Charge Totale, Densité d'énergie moyenne globale}
    return {total_q, total_e / total_volume};
};

// Returns t, Q and E for the specified gradient flow parameters
std::vector<double> mpi::observables::topo_charge_flowed(GaugeField& field, const GeometryCB& geo,
                                                         GradientFlow& gf, mpi::MpiTopology& topo,
                                                         int N_steps_gf, int N_rk_steps) {
    //mpi::exchange::exchange_halos_cascade(field, geo, topo);
    std::vector<double> tQE(3 * N_steps_gf);
    gf.copy(field);
    double t = 0.0;
    int p = 6;
    int pt = 2;
    for (int steps = 0; steps < N_steps_gf; steps++) {
        auto qe = topo_q_e_clover_global(gf.field_c, geo, topo);
        tQE[3 * steps] = t;
        tQE[3 * steps + 1] = qe.first;
        tQE[3 * steps + 2] = qe.second;
        if (topo.rank == 0) {
            std::cout << "n = " << steps << ", t = " << io::format_double(t, pt)
                      << ", Q = " << io::format_double(qe.first, p)
                      << ", t²E = " << io::format_double(t * t * qe.second, p) << "\n";
        }
        for (int rk_steps = 0; rk_steps < N_rk_steps; rk_steps++) {
            gf.rk3_step(topo);
            t += gf.epsilon;
        }
    }
    return tQE;
};
