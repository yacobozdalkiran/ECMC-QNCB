//
// Created by ozdalkiran-l on 1/29/26.
//

#ifndef ECMC_MPI_GRADIENT_FLOW_H
#define ECMC_MPI_GRADIENT_FLOW_H

#include "../gauge/GaugeField.h"
#include "../mpi/MpiTopology.h"

class GradientFlow {
public:
    double epsilon;
    GaugeField field_c;
    GaugeField force_0;
    GaugeField force_1;
    const GeometryCB* geo_p;

    explicit GradientFlow(double epsilon_, const GaugeField &field, const GeometryCB &geo);
    void copy(const GaugeField& field);
    void compute_force(int i);
    void compute_w1();
    void compute_w2();
    void replace_force_0();
    void compute_w3();
    void rk3_step(mpi::MpiTopology &topo);
};

#endif //ECMC_MPI_GRADIENT_FLOW_H
