#pragma once
#include "../gauge/GaugeField.h"
#include "HalosShift.h"
#include "MpiTopology.h"

namespace mpi::shift {
void pack_shift_slice(const GaugeField& field, const GeometryCB& geo, std::vector<Complex>& buffer,
                      int dim, int start, int end);
void unpack_shift_slice(GaugeField& field, const GeometryCB& geo,
                        const std::vector<Complex>& buffer, int dim, int start, int end);
void apply_local_shift(GaugeField& field, const GeometryCB& geo, int dim, int s);
void shift_field(GaugeField& field, const GeometryCB& geo, HalosShift& h, mpi::MpiTopology& topo,
                 const int s[4]);
void random_shift(GaugeField& field, const GeometryCB& geo, HalosShift& h, mpi::MpiTopology& topo,
                  std::mt19937_64& rng);
};  // namespace mpi::shift
