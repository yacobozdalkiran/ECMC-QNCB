#include <arpa/inet.h>  // Pour htonl/htons (Linux/macOS)

#include <cstring>
#include <string>

#include "../gauge/GaugeField.h"
#include "../mpi/MpiTopology.h"
// Endian swap for doubles (Standard ILDG)
inline void swap_endian_64(void* ptr) {
    uint64_t* val = reinterpret_cast<uint64_t*>(ptr);
    *val = __builtin_bswap64(*val);
}

std::string generate_ildg_xml(int L_glob);
void write_lime_header(MPI_File& fh, MPI_Offset offset, const std::string& type, uint64_t len,
                       bool mb, bool me);
void save_ildg_clime(const std::string& filename, const std::string& dirpath, const GaugeField& field, const GeometryCB& geo,
                     const mpi::MpiTopology& topo);
void read_ildg_clime(const std::string& filename, const std::string& dirpath, GaugeField& field, const GeometryCB& geo,
                     mpi::MpiTopology& topo);
