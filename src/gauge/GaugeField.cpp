//
// Created by ozdalkiran-l on 1/8/26.
//

#include "GaugeField.h"

#include "../su3/utils.h"

// Initialises the gauge field with random SU3 matrices
void GaugeField::hot_start(const GeometryCB& geo, std::mt19937_64& rng) {
    for (int t = 1; t <= T_int; t++)
        for (int z = 1; z <= L_int; z++)
            for (int y = 1; y <= L_int; y++)
                for (int x = 1; x <= L_int; x++) {
                    size_t site = geo.index(x, y, z, t);
                    for (int mu = 0; mu < 4; mu++) {
                        view_link(site, mu) = random_su3(rng);
                    }
                }
}

// Initialises the gauge field with identity matrices
void GaugeField::cold_start(const GeometryCB& geo) {
    for (int t = 1; t <= T_int; t++)
        for (int z = 1; z <= L_int; z++)
            for (int y = 1; y <= L_int; y++)
                for (int x = 1; x <= L_int; x++) {
                    size_t site = geo.index(x, y, z, t);
                    for (int mu = 0; mu < 4; mu++) {
                        view_link(site, mu) = Eigen::Matrix3cd::Identity();
                    }
                }
}

// Projects a link on SU3 using Gramm-Schmidt
void GaugeField::projection_su3(size_t site, int mu) {
    auto U = view_link(site, mu);

    SU3 temp = U;

    Eigen::Vector3cd c0 = temp.col(0);
    c0.normalize();

    Eigen::Vector3cd c1 = temp.col(1);
    c1 -= c0 * c0.dot(c1);
    c1.normalize();

    Eigen::Vector3cd c2;
    c2(0) = std::conj(c0(1) * c1(2) - c0(2) * c1(1));
    c2(1) = std::conj(c0(2) * c1(0) - c0(0) * c1(2));
    c2(2) = std::conj(c0(0) * c1(1) - c0(1) * c1(0));

    temp.col(0) = c0;
    temp.col(1) = c1;
    temp.col(2) = c2;

    U = temp;
}

// Projects the whole field on SU3
void GaugeField::project_field_su3(const GeometryCB& geo) {
    for (int t = 1; t <= T_int; t++)
        for (int z = 1; z <= L_int; z++)
            for (int y = 1; y <= L_int; y++)
                for (int x = 1; x <= L_int; x++) {
                    size_t site = geo.index(x, y, z, t);
                    for (int mu = 0; mu < 4; mu++) {
                        projection_su3(site, mu);
                    }
                }
}

// Computes the sum of all staples of a site
void GaugeField::compute_staple(const GeometryCB& geo, size_t site, int mu, SU3& staple) const {
    staple.setZero();
    size_t x = site;                        // x
    size_t xmu = geo.get_neigh(x, mu, up);  // x+mu
    for (int nu = 0; nu < 4; nu++) {
        if (nu == mu) {
            continue;
        }
        // Staple forward
        size_t xnu = geo.get_neigh(x, nu, up);  // x+nu
        const auto& U0 = view_link_const(xmu, nu);
        const auto& U1 = view_link_const(xnu, mu);
        const auto& U2 = view_link_const(x, nu);
        staple += U0 * (U2 * U1).adjoint();

        // Staple backward
        size_t xmunu = geo.get_neigh(xmu, nu, down);  // x+mu-nu
        size_t xmnu = geo.get_neigh(x, nu, down);     // x-nu
        auto V0 = view_link_const(xmunu, nu);
        auto V1 = view_link_const(xmnu, mu);
        auto V2 = view_link_const(xmnu, nu);
        staple += (V1 * V0).adjoint() * V2;
    }
}

// Performs a random gauge_transform on the field
void GaugeField::gauge_transform(const GeometryCB& geo, std::mt19937_64& rng) {
    GaugeField transform(geo);
    transform.hot_start(geo, rng);
    for (int t = 1; t <= T_int; t++)
        for (int z = 1; z <= L_int; z++)
            for (int y = 1; y <= L_int; y++)
                for (int x = 1; x <= L_int; x++) {
                    size_t site = geo.index(x, y, z, t);
                    for (int mu = 0; mu < 4; mu++) {
                        view_link(site, mu) =
                            transform.view_link_const(site, mu) * view_link_const(site, mu) *
                            transform.view_link_const(geo.get_neigh(site, mu, up), mu).adjoint();
                    }
                }
};
