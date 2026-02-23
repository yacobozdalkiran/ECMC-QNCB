//
// Created by ozdalkiran-l on 1/29/26.
//

#include "gradient_flow.h"

#include "../mpi/HalosExchange.h"
#include "../su3/utils.h"

GradientFlow::GradientFlow(double epsilon_, const GaugeField& field, const GeometryCB& geo)
    : epsilon(epsilon_), field_c(field), force_0(field), force_1(field), geo_p(&geo) {}

// Updates the force field according to staples and links of field_c, into force0 or force1
// depending on i
void GradientFlow::compute_force(int i) {
    SU3 staple{};
    SU3 tmp{};
    for (int t = 1; t <= geo_p->L_int; t++) {
        for (int z = 1; z <= geo_p->L_int; z++) {
            for (int y = 1; y <= geo_p->L_int; y++) {
                for (int x = 1; x <= geo_p->L_int; x++) {
                    size_t site = geo_p->index(x, y, z, t);
                    for (int mu = 0; mu < 4; mu++) {
                        field_c.compute_staple(*geo_p, site, mu, staple);
                        tmp = -1.0 * field_c.view_link_const(site, mu) * staple;
                        proj_lie_su3(tmp);
                        if (i == 0) force_0.view_link(site, mu) = tmp;
                        if (i == 1) force_1.view_link(site, mu) = tmp;
                    }
                }
            }
        }
    }
}

// First RK3 step
void GradientFlow::compute_w1() {
    SU3 tmp{};
    for (int t = 1; t <= geo_p->L_int; t++) {
        for (int z = 1; z <= geo_p->L_int; z++) {
            for (int y = 1; y <= geo_p->L_int; y++) {
                for (int x = 1; x <= geo_p->L_int; x++) {
                    size_t site = geo_p->index(x, y, z, t);
                    for (int mu = 0; mu < 4; mu++) {
                        tmp = exp_su3_luscher(force_0.view_link_const(site, mu), 0.25 * epsilon) *
                              field_c.view_link_const(site, mu);
                        field_c.view_link(site, mu) = tmp;
                    }
                }
            }
        }
    }
}

// Second RK3 step
void GradientFlow::compute_w2() {
    SU3 tmp{};
    SU3 Z{};
    for (int t = 1; t <= geo_p->L_int; t++) {
        for (int z = 1; z <= geo_p->L_int; z++) {
            for (int y = 1; y <= geo_p->L_int; y++) {
                for (int x = 1; x <= geo_p->L_int; x++) {
                    size_t site = geo_p->index(x, y, z, t);
                    for (int mu = 0; mu < 4; mu++) {
                        Z = (8.0 / 9.0) * epsilon * force_1.view_link_const(site, mu) -
                            (17.0 / 36.0) * epsilon * force_0.view_link_const(site, mu);
                        tmp = exp_su3_luscher(Z, 1.0) * field_c.view_link_const(site, mu);
                        field_c.view_link(site, mu) = tmp;
                    }
                }
            }
        }
    }
}

// Replacement Z0 = 8/9 Z_1 -17/36 Z0 for w3
void GradientFlow::replace_force_0() {
    SU3 tmp{};
    for (int t = 1; t <= geo_p->L_int; t++) {
        for (int z = 1; z <= geo_p->L_int; z++) {
            for (int y = 1; y <= geo_p->L_int; y++) {
                for (int x = 1; x <= geo_p->L_int; x++) {
                    size_t site = geo_p->index(x, y, z, t);
                    for (int mu = 0; mu < 4; mu++) {
                        tmp = 8.0 / 9.0 * force_1.view_link_const(site, mu) -
                              (17.0 / 36.0) * force_0.view_link_const(site, mu);
                        force_0.view_link(site, mu) = tmp;
                    }
                }
            }
        }
    }
}

// Third RK3 step
void GradientFlow::compute_w3() {
    SU3 tmp{};
    SU3 Z{};
    for (int t = 1; t <= geo_p->L_int; t++) {
        for (int z = 1; z <= geo_p->L_int; z++) {
            for (int y = 1; y <= geo_p->L_int; y++) {
                for (int x = 1; x <= geo_p->L_int; x++) {
                    size_t site = geo_p->index(x, y, z, t);
                    for (int mu = 0; mu < 4; mu++) {
                        Z = 0.75 * epsilon * force_1.view_link_const(site, mu) -
                            epsilon * force_0.view_link_const(site, mu);
                        tmp = exp_su3_luscher(Z, 1.0) * field_c.view_link_const(site, mu);
                        field_c.view_link(site, mu) = tmp;
                    }
                }
            }
        }
    }
}

// Performs a full RK3 step, the field updated of epsilon with correct halos is in field_c
void GradientFlow::rk3_step(mpi::MpiTopology &topo) {
    compute_force(0);
    compute_w1();
    mpi::exchange::exchange_halos_cascade(field_c, *geo_p, topo);
    compute_force(1);
    compute_w2();
    mpi::exchange::exchange_halos_cascade(field_c, *geo_p, topo);
    replace_force_0();
    compute_force(1);
    compute_w3();
    mpi::exchange::exchange_halos_cascade(field_c, *geo_p, topo);
}

//Copies the content of field into field_c
void GradientFlow::copy(const GaugeField& field) {
    field_c = field;
};
