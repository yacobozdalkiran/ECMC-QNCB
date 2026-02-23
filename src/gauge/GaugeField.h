//
// Created by ozdalkiran-l on 1/8/26.
//

#ifndef INC_4D_MPI_GAUGEFIELD_H
#define INC_4D_MPI_GAUGEFIELD_H

#include <Eigen/Dense>
#include <complex>
#include <random>
#include <vector>

#include "../geometry/GeometryCB.h"

using Complex = std::complex<double>;
using SU3 = Eigen::Matrix<std::complex<double>, 3, 3, Eigen::RowMajor>;

class GaugeField {
    int L_ext;
    int L_int;
    int T_ext;
    int T_int;
    size_t V_ext;
    size_t V_int;

public:
    std::vector<Complex> links;

    // Initializes a squared cold gauge conf with halos
    explicit GaugeField(const GeometryCB& geo)
        : L_ext(geo.L_ext),
          L_int(geo.L_int),
          T_ext(geo.L_ext),
          T_int(geo.L_int),
          V_ext(geo.V_ext),
          V_int(geo.V_int),
          links(V_ext * 4 * 9, Complex(0.0, 0.0)) {
        for (size_t site = 0; site < V_ext; site++) {
            for (int mu = 0; mu < 4; mu++) {
                view_link(site, mu) = SU3::Identity();
            }
        }
    }

    void hot_start(const GeometryCB& geo, std::mt19937_64& rng);
    void cold_start(const GeometryCB& geo);

    // Non const mapping of links to SU3 matrices
    Eigen::Map<SU3> view_link(size_t site, int mu) {
        return Eigen::Map<SU3>(&links[(site * 4 + mu) * 9]);
    }

    // Const mapping of links to SU3 matrices
    [[nodiscard]] Eigen::Map<const SU3> view_link_const(size_t site, int mu) const {
        return Eigen::Map<const SU3>(&links[(site * 4 + mu) * 9]);
    }

    void projection_su3(size_t site, int mu);
    void project_field_su3(const GeometryCB &geo);
    void compute_staple(const GeometryCB &geo, size_t site, int mu, SU3 &staple) const;
    void gauge_transform(const GeometryCB& geo, std::mt19937_64& rng);
};

#endif  // INC_4D_MPI_GAUGEFIELD_H
