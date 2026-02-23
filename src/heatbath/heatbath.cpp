//
// Created by ozdalkiran-l on 1/21/26.
//

#include "heatbath.h"

// Generates a SU2 Heatbath candidate
SU2q heatbath::generate_su2_kp(double beta, double k, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist(0, 1);
    double alpha = beta * k * 2.0 / 3.0;

    // Generate a0 according to exp(alpha * a0) * sqrt(1 - a0^2)
    double a0;
    while (true) {
        double delta = dist(rng);
        double r1 = dist(rng);
        double r2 = dist(rng);
        double r3 = dist(rng);

        // Kennedy-Pendleton for a0
        double lambda_sq = -0.5 / alpha * (log(r1) + pow(cos(2 * M_PI * r2), 2) * log(r3));
        if (delta * delta <= 1.0 - lambda_sq) {
            a0 = 1.0 - 2.0 * lambda_sq;
            break;
        }
    }

    // Uniform generation on 3d sphere for the other coordinates
    double r_spatial = sqrt(1.0 - a0 * a0);
    double phi = 2.0 * M_PI * dist(rng);
    double cos_theta = 2.0 * dist(rng) - 1.0;
    double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    return {a0, r_spatial * sin_theta * cos(phi), r_spatial * sin_theta * sin(phi),
            r_spatial * cos_theta};
}

// Generates r such that dP(r) = dr*exp(beta/3Tr(rw)) and w = kv with v in SU2 and k = sqrt(det(w))
SU2q heatbath::su2_step(double beta, const SU2q& w, std::mt19937_64& rng) {
    double k = w.norm();
    SU2q v = w / k;
    SU2q x = generate_su2_kp(beta, k, rng);
    return mult(x, adj(v));
}

// Generates a heatbath hit
void heatbath::hit(GaugeField& field, const GeometryCB& geo, size_t site, int mu, double beta,
                   SU3& A, std::mt19937_64& rng) {
    field.compute_staple(geo, site, mu, A);

    SU3 W = field.view_link_const(site, mu) * A;
    SU2q wq = su2_to_quaternion(W.block<2, 2>(0, 0));
    SU2q r = su2_step(beta, wq, rng);
    SU3 R = su2_quaternion_to_su3(r, 0, 1);
    field.view_link(site, mu) = R * field.view_link(site, mu);

    W = field.view_link_const(site, mu) * A;
    wq = su2_to_quaternion(W.block<2, 2>(1, 1));
    r = su2_step(beta, wq, rng);
    R = su2_quaternion_to_su3(r, 1, 2);
    field.view_link(site, mu) = R * field.view_link(site, mu);

    W = field.view_link_const(site, mu) * A;
    SU2 wsu2;
    wsu2 << W(0, 0), W(0, 2), W(2, 0), W(2, 2);
    wq = su2_to_quaternion(wsu2);
    r = su2_step(beta, wq, rng);
    R = su2_quaternion_to_su3(r, 0, 2);
    field.view_link(site, mu) = R * field.view_link(site, mu);
}

// Full heatbath sweep with N_hits hits
void heatbath::sweep(GaugeField& field, const GeometryCB& geo, double beta, int N_hits,
                     std::mt19937_64& rng) {
    SU3 A;
    for (int t = 1; t <= geo.L_int; t++) {
        for (int z = 1; z <= geo.L_int; z++) {
            for (int y = 1; y <= geo.L_int; y++) {
                for (int x = 1; x <= geo.L_int; x++) {
                    size_t site = geo.index(x, y, z, t);
                    for (int mu = 0; mu < 4; mu++) {
                        for (int h = 0; h < N_hits; h++) {
                            hit(field, geo, site, mu, beta, A, rng);
                        }
                    }
                }
            }
        }
    }
}
