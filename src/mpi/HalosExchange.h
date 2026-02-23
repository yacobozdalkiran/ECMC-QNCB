#pragma once

#include "../gauge/GaugeField.h"
#include "MpiTopology.h"

namespace mpi::exchange {
void pack_face(const GaugeField& field, const GeometryCB& geo, std::vector<Complex>& buffer,
               int dim, int face_coord, const int min_c[4], const int max_c[4]);
void unpack_face(GaugeField& field, const GeometryCB& geo, const std::vector<Complex>& buffer,
                 int dim, int face_coord, const int min_c[4], const int max_c[4]);
void exchange_halos_cascade(GaugeField& field, const GeometryCB& geo, mpi::MpiTopology& topo);
}  // namespace mpi::exchange
