//
//
// Created by ozdalkiran-l on 1/14/26.
//

#ifndef ECMC_MPI_IO_H
#define ECMC_MPI_IO_H

#include <random>
#include <vector>

#include "../mpi/MpiTopology.h"
#include "params.h"
#include "../ecmc/ecmc_mpi_cb.h"

namespace io {
// Output
void save_plaquette(const std::vector<double>& data, const std::string& filename, const std::string& dirpath, int precision);
void save_event_nb(const std::vector<size_t>& event_nb, const std::string& filename, const std::string& dirpath);
void save_topo(const std::vector<double>& tQE, const std::string& filename, const std::string& dirpath, int precision);
void save_seed(std::mt19937_64& rng, const std::string& filename, const std::string& dirpath, mpi::MpiTopology& topo);
void save_params(const RunParamsHbCB& rp, const std::string& filename, const std::string& dirpath);
void save_params(const RunParamsECB& rp, const std::string& filename, const std::string& dirpath);
void add_shift(int shift, const std::string& filename, const std::string& dirpath);
void add_finished(const std::string& filename, const std::string& dirpath);
void save_state(const LocalChainState& state, const std::string& filename,
                const std::string& dirpath, mpi::MpiTopology& topo);
// Input
std::string trim(const std::string& s);
void load_params(const std::string& filename, RunParamsECB& rp);
void load_params(const std::string& filename, RunParamsHbCB& rp);
void load_state(LocalChainState& state, const std::string& filename,
                    const std::string& dirpath, mpi::MpiTopology& topo);


// Utilitaries
std::string format_double(double val, int precision);

//Load parameters from input file
bool read_params(RunParamsHbCB& params, int rank, const std::string& input);
bool read_params(RunParamsECB& params, int rank, const std::string& input);
}  // namespace io

// Printing
void print_parameters(const RunParamsHbCB& rp, const mpi::MpiTopology& topo);
void print_parameters(const RunParamsECB& rp, const mpi::MpiTopology& topo);
void print_time(long elapsed);

#endif  // ECMC_MPI_IO_H
