#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "../flow/gradient_flow.h"
#include "../gauge/GaugeField.h"
#include "../heatbath/heatbath_mpi.h"
#include "../io/ildg.h"
#include "../io/io.h"
#include "../mpi/HalosExchange.h"
#include "../mpi/HalosShift.h"
#include "../mpi/Shift.h"
#include "../observables/observables_mpi.h"

namespace fs = std::filesystem;

void generate_hb_cb(const RunParamsHbCB& rp, bool existing) {
    //========================Objects initialization====================
    // MPI
    int n_core_dims = rp.n_core_dims;
    mpi::MpiTopology topo(n_core_dims);

    // Lattice creation + RNG
    int L = rp.L_core;
    GeometryCB geo(L);
    GaugeField field(geo);
    std::mt19937_64 rng(rp.seed + topo.rank);
    if (!rp.cold_start) {
        field.hot_start(geo, rng);
    }

    if (existing) {
        read_ildg_clime(rp.run_name, rp.run_dir, field, geo, topo);
        fs::path state_path = fs::path(rp.run_dir) / rp.run_name / (rp.run_name + "_seed") /
                              (rp.run_name + "_seed" + std::to_string(topo.rank) + ".txt");
        std::ifstream ifs(state_path);
        if (ifs.is_open()) {
            ifs >> rng;  // On ignore la seed initiale, on reprend l'état exactement
        } else {
            std::cerr << "Could not open " << state_path << "\n";
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // Initalization of halos
    mpi::exchange::exchange_halos_cascade(field, geo, topo);

    // Params HB
    HbParams hp = rp.hp;

    // Shift objects
    HalosShift halo_shift(geo);

    int N_shift = rp.N_shift;
    int N_switch_eo = rp.N_switch_eo;

    // Topo
    double eps = 0.02;
    GradientFlow flow(eps, field, geo);

    // Measure vectors
    std::vector<double> plaquette;
    plaquette.reserve(rp.save_each_shifts / rp.N_shift_plaquette + 2);

    std::vector<double> tQE_tot;
    std::vector<double> tQE_current;
    if (rp.topo) {
        tQE_tot.reserve(3 * (rp.save_each_shifts / rp.N_shift_topo + 2) * rp.N_rk_steps *
                        rp.N_steps_gf);
        tQE_current.reserve(3 * rp.N_rk_steps * rp.N_steps_gf);
    }

    // Save params
    if (topo.rank == 0) {
        io::save_params(rp, rp.run_name, rp.run_dir);
    }

    // Print params
    print_parameters(rp, topo);

    //==============================Heatbath Checkboard===========================

    // Thermalisation

    if (topo.rank == 0) {
        std::cout << "\n\n===========================================\n";
        std::cout << "Thermalisation : " << rp.N_therm << " shifts\n";
        std::cout << "===========================================\n";
    }
    for (int i = 0; i < rp.N_therm; i++) {
        // Random shift
        mpi::shift::random_shift(field, geo, halo_shift, topo, rng);
        // HB
        mpi::heatbathcb::samples(field, geo, topo, hp, rng);
        // Halos update
        mpi::exchange::exchange_halos_cascade(field, geo, topo);

        // Plaquette measure (not saved for thermalization)
        if (i % rp.N_shift_plaquette == 0) {
            double p = mpi::observables::mean_plaquette_global(field, geo, topo);
            if (topo.rank == 0) {
                std::cout << "====== Plaquette ======\n";
                std::cout << "(Therm) Sample " << i / rp.N_shift_plaquette << ", <P> = " << p
                          << "\n";
            }
        }
    }

    // Sampling
    if (topo.rank == 0) {
        std::cout << "\n\n===========================================\n";
        std::cout << "Sampling : " << rp.N_shift / rp.N_shift_plaquette << " <P> samples, "
                  << rp.N_shift / rp.N_shift_topo << " Q samples\n";
        std::cout << "===========================================\n";
    }

    for (int i = 0; i < N_shift; i++) {
        if (topo.rank == 0) {
            std::cout << "\n\n=============" << "Shift " << i << "=============\n";
        }

        // Random shift
        mpi::shift::random_shift(field, geo, halo_shift, topo, rng);
        // HB
        mpi::heatbathcb::samples(field, geo, topo, hp, rng);
        // Halos update
        mpi::exchange::exchange_halos_cascade(field, geo, topo);

        // Plaquette measure
        if (i % rp.N_shift_plaquette == 0) {
            double p = mpi::observables::mean_plaquette_global(field, geo, topo);
            if (topo.rank == 0) {
                std::cout << "====== Plaquette ======\n";
                std::cout << "Sample " << i / rp.N_shift_plaquette << ", <P> = " << p << "\n";
            }
            plaquette.emplace_back(p);
        }
        // Measure topo
        if (rp.topo and (i % rp.N_shift_topo == 0)) {
            if (topo.rank == 0) {
                std::cout << "====== Topology ======\n";
            }
            tQE_current = mpi::observables::topo_charge_flowed(field, geo, flow, topo,
                                                               rp.N_steps_gf, rp.N_rk_steps);
            if (topo.rank == 0) {
                std::cout << "Sample " << i / rp.N_shift_topo
                          << ", final Q = " << tQE_current[tQE_current.size() - 2]
                          << "\n";  // Print the last value of Q
            }
            tQE_tot.insert(tQE_tot.end(), std::make_move_iterator(tQE_current.begin()),
                           std::make_move_iterator(tQE_current.end()));
        }

        // Save conf/seed/obs
        if (i > 0 and i % rp.save_each_shifts == 0) {
            if (topo.rank == 0) {
                std::cout << "\n\n==========================================\n";
                // Write the output
                int precision = 10;
                io::save_plaquette(plaquette, rp.run_name, rp.run_dir, precision);
                if (rp.topo) {
                    io::save_topo(tQE_tot, rp.run_name, rp.run_dir, precision);
                }
                io::add_shift(i, rp.run_name, rp.run_dir);
            }
            // Save seeds
            io::save_seed(rng, rp.run_name, rp.run_dir, topo);
            // Save conf
            save_ildg_clime(rp.run_name, rp.run_dir, field, geo, topo);
            if (topo.rank == 0) {
                std::cout << "==========================================\n";
            }
            // Clear the measures
            plaquette.clear();
            plaquette.reserve(rp.save_each_shifts / rp.N_shift_plaquette + 2);
            tQE_tot.clear();
            tQE_tot.reserve(3 * (rp.save_each_shifts / rp.N_shift_topo + 2) * rp.N_rk_steps *
                            rp.N_steps_gf);
        }
    }

    //===========================Output======================================

    if (topo.rank == 0) {
        std::cout << "\n\n==========================================\n";
        // Write the output
        int precision = 10;
        io::save_plaquette(plaquette, rp.run_name, rp.run_dir, precision);
        if (rp.topo) {
            io::save_topo(tQE_tot, rp.run_name, rp.run_dir, precision);
        }
        io::add_shift(rp.N_shift, rp.run_name, rp.run_dir);
        io::add_finished(rp.run_name, rp.run_dir);
    }
    // Save seeds
    io::save_seed(rng, rp.run_name, rp.run_dir, topo);
    // Save conf
    save_ildg_clime(rp.run_name, rp.run_dir, field, geo, topo);
    if (topo.rank == 0) {
        std::cout << "==========================================\n";
    }
}

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Trying to read input
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (argc < 2) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <input_file.txt>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Charging the parameters of the run
    RunParamsHbCB params;
    bool existing = io::read_params(params, rank, argv[1]);

    // Measuring time
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    generate_hb_cb(params, existing);

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if (rank == 0) {
        double total_time = end_time - start_time;
        std::cout << std::fixed << std::setprecision(4);
        print_time(total_time);
    }
    // End MPI
    MPI_Finalize();
}
