//
// Created by ozdalkiran-l on 1/8/26.
//

#include "utils.h"

#include <iostream>
#include <unsupported/Eigen/MatrixFunctions>

#include "../gauge/GaugeField.h"

// Generates a random SU3 matrix using QR decomposition
SU3 random_su3(std::mt19937_64& rng) {
    std::normal_distribution<double> gauss(0.0, 1.0);
    Eigen::Matrix3cd z;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            z(i, j) = Complex(gauss(rng), gauss(rng));
        }
    }

    // QR Decomposition
    Eigen::HouseholderQR<SU3> qr(z);
    SU3 Q = qr.householderQ();

    Complex detQ = Q.determinant();
    Q /= std::pow(detQ, 1.0 / 3.0);

    return Q;
}

// Embbeds a SU2 matrix into a SU3 matrix according to subgroup (i,j)
SU3 su2_quaternion_to_su3(const SU2q& su2, int i, int j) {
    if (i == j) std::cerr << "i = j wrong embedding\n";
    SU3 X;
    int k = 3 - i - j;
    X.setZero();
    X(k, k) = Complex(1.0, 0.0);
    X(i, i) = Complex(su2(0), su2(3));
    X(j, j) = Complex(su2(0), -su2(3));
    X(i, j) = Complex(su2(2), su2(1));
    X(j, i) = Complex(-su2(2), su2(1));
    return X;
}

// Returns the quat. representation of SU2 matrix
SU2q su2_to_quaternion(const SU2& su2) {
    return {(su2(0, 0) + su2(1, 1)).real() / 2.0, (su2(0, 1) + su2(1, 0)).imag() / 2.0,
            (su2(0, 1) - su2(1, 0)).real() / 2.0, (su2(0, 0) - su2(1, 1)).imag() / 2.0};
}

// Multiplication of 2 SU2q
SU2q mult(const SU2q& q, const SU2q& p) {
    return {
        q(0) * p(0) - q(1) * p(1) - q(2) * p(2) - q(3) * p(3),  // r0
        q(0) * p(1) + q(1) * p(0) + q(2) * p(3) - q(3) * p(2),  // r1 (i)
        q(0) * p(2) - q(1) * p(3) + q(2) * p(0) + q(3) * p(1),  // r2 (j)
        q(0) * p(3) + q(1) * p(2) - q(2) * p(1) + q(3) * p(0)   // r3 (k)
    };
}

// Returns the adjoint
SU2q adj(const SU2q& q) { return {q(0), -q(1), -q(2), -q(3)}; }

// Returns a random SU3 matrix epsilon close to identity
SU3 random_SU3_epsilon(double epsilon, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> unif(-0.5, 0.5);
    SU2q x = {0.0, 0.0, 0.0, 0.0};
    SU3 M = SU3::Identity();

    // double r0 = unif(rng);
    double r1 = unif(rng);
    double r2 = unif(rng);
    double r3 = unif(rng);
    double norm = sqrt(r1 * r1 + r2 * r2 + r3 * r3);
    x(0) = sqrt(1 - epsilon * epsilon);
    x(1) = epsilon * r1 / norm;
    x(2) = epsilon * r2 / norm;
    x(3) = epsilon * r3 / norm;
    M *= su2_quaternion_to_su3(x, 0, 1);

    // r0 = unif(rng);
    r1 = unif(rng);
    r2 = unif(rng);
    r3 = unif(rng);
    norm = sqrt(r1 * r1 + r2 * r2 + r3 * r3);
    x(0) = sqrt(1 - epsilon * epsilon);
    x(1) = epsilon * r1 / norm;
    x(2) = epsilon * r2 / norm;
    x(3) = epsilon * r3 / norm;
    M *= su2_quaternion_to_su3(x, 0, 2);

    // r0 = unif(rng);
    r1 = unif(rng);
    r2 = unif(rng);
    r3 = unif(rng);
    norm = sqrt(r1 * r1 + r2 * r2 + r3 * r3);
    x(0) = sqrt(1 - epsilon * epsilon);
    x(1) = epsilon * r1 / norm;
    x(2) = epsilon * r2 / norm;
    x(3) = epsilon * r3 / norm;
    M *= su2_quaternion_to_su3(x, 1, 2);

    return M;
}

// Creates a symmetric set of SU3 matrices close to identity
std::vector<SU3> metropolis_set(double epsilon, int size, std::mt19937_64& rng) {
    std::vector<SU3> set(size + 1);
    set[0] = SU3::Identity();
    for (int i = 1; i < size + 1; i += 2) {
        set[i] = random_SU3_epsilon(epsilon, rng);
        set[i + 1] = set[i].adjoint();
    }
    return set;
}

// Creates a set of SU3 matrices close to identity
std::vector<SU3> ecmc_set(double epsilon, std::vector<SU3>& set, std::mt19937_64& rng) {
    size_t size = set.size() - 1;
    set[0] = SU3::Identity();
    for (size_t i = 1; i < size + 1; i += 2) {
        set[i] = random_SU3_epsilon(epsilon, rng);
        set[i + 1] = set[i].adjoint();
    }
    return set;
}

// Projects A on the lie algebra su(3) (traceless anti-hermitian)
void proj_lie_su3(SU3& A) {
    SU3 B = (0.5 * (A - A.adjoint())).eval();  // To avoid aliasing
    std::complex<double> tr_per_dim = B.trace() / 3.0;
    A = B - tr_per_dim * SU3::Identity();
}

// Returns the exponential of a lie algebra su(3) matrix Z
// Eigen impl for prototyping at first
SU3 exp_analytic(const SU3& Z, double coeff) { return (Z * coeff).exp(); }

/**
 * Calcule l'exponentielle d'une matrice X appartenant à l'algèbre su(3).
 * X doit être anti-hermitienne et de trace nulle.
 * La méthode repose sur le schéma de Horner de Lüscher (Cayley-Hamilton).
 */
SU3 exp_su3_luscher(const SU3& Z, double coeff) {
    using namespace std::complex_literals;
    SU3 X = coeff*Z;
    // 1. Calcul des invariants invariants t et d
    // t = -1/2 * tr(X^2)
    // d = i * det(X)
    double t = -0.5 * (X * X).trace().real();
    double d = (1.0i * X.determinant()).real();
    if (t > 1.0) {
        std::cerr << "Warning : ||X|| > 1, unstable exp\n";
    }
    // 2. Choix du degré N du polynôme [cite: 361]
    // Pour une précision double (DBL_EPSILON), N=18 est suffisant.
    const int N = 18;

    // 3. Calcul des coefficients de Taylor c_n = 1/n! [cite: 343]
    // On peut les pré-calculer ou les calculer à la volée.
    double c[N + 1];
    c[0] = 1.0;
    for (int n = 1; n <= N; ++n) {
        c[n] = c[n - 1] / static_cast<double>(n);
    }

    // 4. Schéma de récurrence (Horner) [cite: 348, 349]
    // q_n = q_{n,0} + q_{n,1}*X + q_{n,2}*X^2
    std::complex<double> q0, q1, q2;

    // Initialisation au rang N
    q0 = c[N];
    q1 = 0.0;
    q2 = 0.0;

    // Boucle de N-1 à 0 [cite: 349]
    for (int n = N - 1; n >= 0; --n) {
        std::complex<double> q0_next = c[n] - 1.0i * d * q2;
        std::complex<double> q1_next = q0 - t * q2;
        std::complex<double> q2_next = q1;

        q0 = q0_next;
        q1 = q1_next;
        q2 = q2_next;
    }

    // 5. Reconstruction de la matrice finale exp(X) = q0*I + q1*X + q2*X^2 [cite: 327, 347]
    SU3 Id = SU3::Identity();
    SU3 X2 = X * X;

    return q0 * Id + q1 * X + q2 * X2;
}
