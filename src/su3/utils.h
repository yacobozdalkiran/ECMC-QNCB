//
// Created by ozdalkiran-l on 1/8/26.
//

#ifndef INC_4D_MPI_UTILS_H
#define INC_4D_MPI_UTILS_H

#include <Eigen/Dense>
#include <complex>
#include <random>

using SU3 = Eigen::Matrix<std::complex<double>, 3, 3, Eigen::RowMajor>;
using Complex = std::complex<double>;
using SU2q = Eigen::Vector4d;
using SU2 = Eigen::Matrix2cd;

// Generates an exp(i xi lambda_3)
inline SU3 el_3(double xi) {
    SU3 result;
    double cxi = cos(xi);
    double sxi = sin(xi);
    result << Complex(cxi, sxi), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0),
        Complex(cxi, -sxi), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0),
        Complex(1.0, 0.0);
    return result;
}
SU3 random_su3(std::mt19937_64& rng);
SU3 su2_quaternion_to_su3(const SU2q& su2, int i, int j);
SU2q su2_to_quaternion(const SU2& su2);
SU2q mult(const SU2q& q, const SU2q& p);
SU2q adj(const SU2q& q);
SU3 random_SU3_epsilon(double epsilon, std::mt19937_64& rng);
std::vector<SU3> metropolis_set(double epsilon, int size, std::mt19937_64& rng);
std::vector<SU3> ecmc_set(double epsilon, std::vector<SU3>& set, std::mt19937_64& rng);
// Returns the sign of the double x
inline int dsign(double x) {
    // fonction signe pour double
    if (x > 0) return 1;
    if (x < 0) return -1;
    return 0;
}
void proj_lie_su3(SU3& A);
SU3 exp_analytic(const SU3& Z, double coeff);
SU3 exp_su3_luscher(const SU3& Z, double coeff);
#endif  // INC_4D_MPI_UTILS_H
