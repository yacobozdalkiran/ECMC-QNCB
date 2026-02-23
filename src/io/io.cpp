//
// Created by ozdalkiran-l on 1/14/26.
//

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
extern "C" {
#include <lime.h>
}
#include "io.h"

namespace fs = std::filesystem;

// Saves a vector of doubles in ../data/filename/filename_plaquette.txt
void io::save_plaquette(const std::vector<double>& data, const std::string& filename,
                        const std::string& dirpath, int precision) {
    // Create a data folder if doesn't exists
    fs::path base_dir(dirpath);
    fs::path dir = base_dir / filename;

    try {
        if (!fs::exists(dir)) {
            fs::create_directories(dir);
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Couldn't create folder data : " << e.what() << std::endl;
        return;
    }
    fs::path filepath = dir / (filename + "_plaquette.txt");

    std::ofstream file(filepath, std::ios::out | std::ios::app);
    if (!file.is_open()) {
        std::cout << "Could not open file " << filepath << "\n";
        return;
    }
    file << std::fixed << std::setprecision(precision);
    for (const double& x : data) {
        file << x << "\n";
    }
    file.close();
    std::cout << "Plaquette written in " << filepath << "\n";
}

void io::save_event_nb(const std::vector<size_t>& event_nb, const std::string& filename,
                        const std::string& dirpath) {
    // Create a data folder if doesn't exists
    fs::path base_dir(dirpath);
    fs::path dir = base_dir / filename;

    try {
        if (!fs::exists(dir)) {
            fs::create_directories(dir);
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Couldn't create folder data : " << e.what() << std::endl;
        return;
    }
    fs::path filepath = dir / (filename + "_nbevent.txt");

    std::ofstream file(filepath, std::ios::out | std::ios::app);
    if (!file.is_open()) {
        std::cout << "Could not open file " << filepath << "\n";
        return;
    }
    for (const size_t& x : event_nb) {
        file << x << "\n";
    }
    file.close();
    std::cout << "Number of events written in " << filepath << "\n";
}
// Saves the tQE vector into data/filename/filename_topo.txt
void io::save_topo(const std::vector<double>& tQE, const std::string& filename,
                   const std::string& dirpath, int precision) {
    // Create a data folder if doesn't exists
    fs::path base_dir(dirpath);
    fs::path dir = base_dir / filename;

    try {
        if (!fs::exists(dir)) {
            fs::create_directories(dir);
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Couldn't create folder data : " << e.what() << std::endl;
        return;
    }
    fs::path filepath = dir / (filename + "_topo.txt");

    std::ofstream file(filepath, std::ios::out | std::ios::app);
    if (!file.is_open()) {
        std::cout << "Could not open file " << filepath << "\n";
        return;
    }
    file << std::fixed << std::setprecision(precision);
    for (size_t i = 0; i < tQE.size(); i += 3) {
        file << tQE[i] << " " << tQE[i + 1] << " " << tQE[i + 2] << "\n";
    }
    file.close();
    std::cout << "Topology written in " << filepath << "\n";
};

// Saves the Mersenne Twister states into dirpath/filename/filename_seed/filename_seed[rank].txt
void io::save_seed(std::mt19937_64& rng, const std::string& filename, const std::string& dirpath,
                   mpi::MpiTopology& topo) {
    // Create a data folder if doesn't exists
    fs::path base_dir(dirpath);
    fs::path run_dir =
        base_dir / filename /
        (filename + "_seed");  // Utilise l'opérateur / pour gérer les slashs proprement

    try {
        // create_directories crée dirpath PUIS "data/run_name" si nécessaire
        if (!fs::exists(run_dir)) {
            fs::create_directories(run_dir);
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error creating directory structure " << run_dir << " : " << e.what()
                  << std::endl;
        return;
    }
    fs::path filepath = run_dir / (filename + "_seed" + std::to_string(topo.rank) + ".txt");

    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cout << "Could not open file " << filepath << "\n";
        return;
    }
    file << rng;
    file.close();
    if (topo.rank == 0) {
        std::cout << "Seed saved in " << filepath << "\n";
    }
};

// Saves the local chain state of each core in
// dirpath/filename/filename_state/filename_state[rank].txt
void io::save_state(const LocalChainState& state, const std::string& filename,
                    const std::string& dirpath, mpi::MpiTopology& topo) {
    // 1. Construction du chemin du dossier : dirpath/filename/filename_state/
    fs::path base_dir = fs::path(dirpath) / filename;
    fs::path state_dir = base_dir / (filename + "_state");

    // 2. Création du dossier (le rang 0 s'en occupe)
    if (topo.rank == 0) {
        if (!fs::exists(state_dir)) {
            fs::create_directories(state_dir);
        }
    }
    // On s'assure que le dossier est prêt pour tout le monde
    MPI_Barrier(MPI_COMM_WORLD);

    // 3. Fichier spécifique au rang : filename_state[rank].txt
    std::string file_rank = filename + "_state" + std::to_string(topo.rank) + ".txt";
    fs::path full_path = state_dir / file_rank;

    std::ofstream ofs(full_path);
    if (!ofs.is_open()) {
        std::cerr << "[Rank " << topo.rank
                  << "] Error: Could not open state file for writing: " << full_path << std::endl;
        return;
    }

    // 4. Écriture des données avec précision maximale
    ofs << std::setprecision(17);

    // Scalaires
    ofs << state.site << " " << state.mu << " " << state.epsilon << " "
        << state.theta_parcouru_refresh_site << " " << state.theta_parcouru_refresh_R << " "
        << state.theta_refresh_site << " " << state.theta_refresh_R << " " << state.set_counter
        << " " << state.initialized << "\n";

    // Matrice SU3 R (3x3 nombres complexes)
    // On sauvegarde la partie réelle et imaginaire de chaque composante
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            auto val = state.R(i, j);
            ofs << val.real() << " " << val.imag() << " ";
        }
        ofs << "\n";
    }

    ofs.close();
    if (topo.rank == 0) {
        std::cout << "ECMC Chain saved in : " << full_path << std::endl;
    }
}

// Loads the local chain state of each core
void io::load_state(LocalChainState& state, const std::string& filename, const std::string& dirpath,
                    mpi::MpiTopology& topo) {
    // 1. Construction du chemin du fichier
    fs::path state_path = fs::path(dirpath) / filename / (filename + "_state") /
                          (filename + "_state" + std::to_string(topo.rank) + ".txt");

    // 2. Tentative d'ouverture
    std::ifstream ifs(state_path);
    if (!ifs.is_open()) {
        // Si le fichier n'existe pas, on considère que c'est un nouveau run
        state.initialized = false;
        return;
    }

    // 3. Lecture des scalaires
    ifs >> state.site >> state.mu >> state.epsilon >> state.theta_parcouru_refresh_site >>
        state.theta_parcouru_refresh_R >> state.theta_refresh_site >> state.theta_refresh_R >>
        state.set_counter >> state.initialized;

    // 4. Lecture de la matrice SU3 R
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double re, im;
            ifs >> re >> im;
            // On reconstruit le complexe et on l'assigne à la matrice
            state.R(i, j) = std::complex<double>(re, im);
        }
    }

    ifs.close();

    if (topo.rank == 0) {
        std::cout << "Successfully loaded ECMC chain state from: " << state_path << std::endl;
    }
}

// Saves the run params into data/filename/filename_params.txt
void io::save_params(const RunParamsHbCB& rp, const std::string& filename,
                     const std::string& dirpath) {
    // Create a data/run_name folder if doesn't exists
    fs::path base_dir(dirpath);
    fs::path dir = base_dir / filename;

    try {
        if (!fs::exists(dir)) {
            fs::create_directories(dir);
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Couldn't create folder data : " << e.what() << std::endl;
        return;
    }
    fs::path filepath = dir / (filename + "_params.txt");

    std::ofstream file(filepath, std::ios::out | std::ios::app);
    if (!file.is_open()) {
        std::cout << "Could not open file " << filepath << "\n";
        return;
    }
    file << std::boolalpha;

    file << "\n#########################################################\n\n";

    file << "# Lattice params\n";
    file << "L_core=" << rp.L_core << "\n";
    file << "n_core_dims=" << rp.n_core_dims << "\n";
    file << "cold_start=" << rp.cold_start << "\n\n";

    file << "# Run params\n";
    file << "seed=" << rp.seed << "\n";
    file << "N_shift_therm = " << rp.N_therm << "\n";
    file << "N_shift =" << rp.N_shift << "\n";

    file << "# Heatbath params\n";
    file << "beta = " << rp.hp.beta << "\n";
    file << "N_sweeps = " << rp.hp.N_sweeps << "\n";
    file << "N_hits = " << rp.hp.N_hits << "\n\n";

    file << "# Plaquette params\n";
    file << "N_shift_plaquette = " << rp.N_shift_plaquette << "\n\n";

    file << "# Topo params\n";
    file << "topo = " << rp.topo << "\n";
    file << "N_shift_topo = " << rp.N_shift_topo << "\n";
    file << "N_steps_gf = " << rp.N_steps_gf << "\n";
    file << "N_rk_steps = " << rp.N_rk_steps << "\n\n";

    file << "#Save params\n";
    file << "save_each_shifts = " << rp.save_each_shifts << "\n\n";

    file.close();
    std::cout << "Parameters saved in " << filepath << "\n";
};

// Utilitary function to trim the spaces
std::string io::trim(const std::string& s) {
    size_t first = s.find_first_not_of(" \t");
    if (first == std::string::npos) return "";
    size_t last = s.find_last_not_of(" \t");
    return s.substr(first, (last - first + 1));
}

// Loads the params contained in filename into rp
void io::load_params(const std::string& filename, RunParamsECB& rp) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Impossible d'ouvrir " + filename);

    std::map<std::string, std::string> config;
    std::string line;

    while (std::getline(file, line)) {
        // Ignore comments and empty lines
        if (line.empty() || line[0] == '#') continue;

        std::istringstream is_line(line);
        std::string key, value;
        if (std::getline(is_line, key, '=') && std::getline(is_line, value)) {
            config[trim(key)] = trim(value);
        }
    }

    // Lattice params
    if (config.count("L_core")) rp.L_core = std::stoi(config["L_core"]);
    if (config.count("n_core_dims")) rp.n_core_dims = std::stoi(config["n_core_dims"]);
    if (config.count("cold_start")) rp.cold_start = (config["cold_start"] == "true");
    if (config.count("seed")) rp.seed = std::stoi(config["seed"]);
    if (config.count("N_shift")) rp.N_shift = std::stoi(config["N_shift"]);

    // ECMC params
    if (config.count("beta")) rp.ecmc_params.beta = std::stod(config["beta"]);
    // 1 sample each run bc frozen links increase artificially the correlation
    rp.ecmc_params.N_samples = 1;
    if (config.count("param_theta_sample"))
        rp.ecmc_params.param_theta_sample = std::stod(config["param_theta_sample"]);
    if (config.count("param_theta_refresh_site"))
        rp.ecmc_params.param_theta_refresh_site = std::stod(config["param_theta_refresh_site"]);
    if (config.count("param_theta_refresh_R"))
        rp.ecmc_params.param_theta_refresh_R = std::stod(config["param_theta_refresh_R"]);
    if (config.count("poisson")) rp.ecmc_params.poisson = (config["poisson"] == "true");
    if (config.count("epsilon_set")) rp.ecmc_params.epsilon_set = std::stod(config["epsilon_set"]);

    if (config.count("N_shift_plaquette"))
        rp.N_shift_plaquette = std::stoi(config["N_shift_plaquette"]);
    // Run and topo params
    if (config.count("N_shift_therm")) rp.N_therm = std::stoi(config["N_shift_therm"]);
    if (config.count("topo")) rp.topo = (config["topo"] == "true");
    if (config.count("N_shift_topo")) rp.N_shift_topo = std::stoi(config["N_shift_topo"]);
    if (config.count("N_steps_gf")) rp.N_steps_gf = std::stoi(config["N_steps_gf"]);
    if (config.count("N_rk_steps")) rp.N_rk_steps = std::stoi(config["N_rk_steps"]);
    //
    // Run name
    if (config.count("run_name")) rp.run_name = config["run_name"];
    if (config.count("run_dir")) rp.run_dir = config["run_dir"];
    if (config.count("save_each_shifts"))
        rp.save_each_shifts = std::stoi(config["save_each_shifts"]);
}

// Loads the params contained in filename into rp
void io::load_params(const std::string& filename, RunParamsHbCB& rp) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Can't open file " + filename);

    std::map<std::string, std::string> config;
    std::string line;

    while (std::getline(file, line)) {
        // Ignore comments and empty lines
        if (line.empty() || line[0] == '#') continue;

        std::istringstream is_line(line);
        std::string key, value;
        if (std::getline(is_line, key, '=') && std::getline(is_line, value)) {
            config[trim(key)] = trim(value);
        }
    }

    // Lattice params
    if (config.count("L_core")) rp.L_core = std::stoi(config["L_core"]);
    if (config.count("n_core_dims")) rp.n_core_dims = std::stoi(config["n_core_dims"]);
    if (config.count("cold_start")) rp.cold_start = (config["cold_start"] == "true");
    if (config.count("seed")) rp.seed = std::stoi(config["seed"]);
    if (config.count("N_shift")) rp.N_shift = std::stoi(config["N_shift"]);

    // Hb params
    if (config.count("beta")) rp.hp.beta = std::stod(config["beta"]);
    // 1 sample bc frozen links increase artificially the correlation
    rp.hp.N_samples = 1;
    if (config.count("N_sweeps")) rp.hp.N_sweeps = std::stoi(config["N_sweeps"]);
    if (config.count("N_hits")) rp.hp.N_hits = std::stoi(config["N_hits"]);
    if (config.count("N_shift_plaquette"))
        rp.N_shift_plaquette = std::stoi(config["N_shift_plaquette"]);

    // Run and topo params
    if (config.count("N_shift_therm")) rp.N_therm = std::stoi(config["N_shift_therm"]);
    if (config.count("topo")) rp.topo = (config["topo"] == "true");
    if (config.count("N_shift_topo")) rp.N_shift_topo = std::stoi(config["N_shift_topo"]);
    if (config.count("N_steps_gf")) rp.N_steps_gf = std::stoi(config["N_steps_gf"]);
    if (config.count("N_rk_steps")) rp.N_rk_steps = std::stoi(config["N_rk_steps"]);

    // Run name
    if (config.count("run_name")) rp.run_name = config["run_name"];
    if (config.count("run_dir")) rp.run_dir = config["run_dir"];
    if (config.count("save_each_shifts"))
        rp.save_each_shifts = std::stoi(config["save_each_shifts"]);
}

// Print parameters of the run
void print_parameters(const RunParamsHbCB& rp, const mpi::MpiTopology& topo) {
    if (topo.rank == 0) {
        std::cout << "==========================================" << std::endl;
        std::cout << "Heatbath - MPI No Checkboard" << std::endl;
        std::cout << "==========================================" << std::endl;
        std::cout << "Run name : " + rp.run_name << "\n";
        std::cout << "---Lattice params---\n";
        std::cout << "Total lattice size : " << rp.L_core * rp.n_core_dims << "^4\n";
        std::cout << "Local lattice size : " << rp.L_core << "^4\n\n";

        std::cout << "---Run params---\n";
        std::cout << "Initial seed : " << rp.seed << "\n";
        std::cout << "Thermalization shifts : " << rp.N_therm << "\n";
        std::cout << "Number of shifts : " << rp.N_shift << "\n";
        std::cout << "Save each : " << rp.save_each_shifts << " shifts\n\n";

        std::cout << "---Heatbath params---\n";
        std::cout << "Beta : " << rp.hp.beta << "\n";
        std::cout << "Number of sweeps : " << rp.hp.N_sweeps << "\n";
        std::cout << "Number of hits : " << rp.hp.N_hits << "\n\n";

        std::cout << "---Measures params---\n";
        std::cout << "Number of <P> samples : " << rp.N_shift / rp.N_shift_plaquette << "\n";
        std::cout << "Measure <P> each : " << rp.N_shift_plaquette << " shifts\n";
        std::cout << "Measure topo : " << (rp.topo ? "Yes" : "No") << "\n";
        if (rp.topo) {
            std::cout << "Number of Q samples : " << rp.N_shift / rp.N_shift_topo << "\n";
            std::cout << "Measure Q each : " << rp.N_shift_topo << " shifts\n\n";
            std::cout << "---Gradient Flow---\n";
            std::cout << "Number of RK3 steps per GF step : " << rp.N_rk_steps << "\n";
            std::cout << "Number of GF steps : " << rp.N_steps_gf << "\n";
        }
        std::cout << "==========================================" << std::endl;
    }
}

// Print time
void print_time(long elapsed) {
    std::cout << "==========================================" << std::endl;
    std::cout << "Elapsed time : " << elapsed << "s\n";
    std::cout << "==========================================" << std::endl;
}

// to_string for double with a fixed precision
std::string io::format_double(double val, int precision) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(precision) << val;
    return ss.str();
}

// Reads the parameters of input file into RunParams struct
// Returns true if the necessary files are found
bool io::read_params(RunParamsHbCB& params, int rank, const std::string& input) {
    if (rank == 0) {
        try {
            io::load_params(input, params);
        } catch (const std::exception& e) {
            std::cerr << "Error reading input : " << e.what() << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    // 1. Diffusion des paramètres numériques (Lattice + Run + Topo)
    // On diffuse les blocs un par un pour plus de clarté et de sécurité
    MPI_Bcast(&params.L_core, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.n_core_dims, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.cold_start, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.N_shift, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.N_therm, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.topo, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.N_shift_topo, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.N_shift_plaquette, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.N_steps_gf, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.N_rk_steps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.save_each_shifts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(&params.hp.beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.hp.N_samples, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.hp.N_hits, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.hp.N_sweeps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.hp.N_therm, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 3. Diffusion de la std::string (run_name)
    int name_len;
    if (rank == 0) {
        name_len = static_cast<int>(params.run_name.size());
    }

    // On envoie d'abord la taille de la string
    MPI_Bcast(&name_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Les autres ranks préparent leur mémoire
    if (rank != 0) {
        params.run_name.resize(name_len);
    }

    // On envoie le contenu de la string (le buffer interne)
    if (name_len > 0) {
        MPI_Bcast(&params.run_name[0], name_len, MPI_CHAR, 0, MPI_COMM_WORLD);
    }
    // 3. Diffusion de la std::string (run_dir)
    int dir_len;
    if (rank == 0) {
        dir_len = static_cast<int>(params.run_dir.size());
    }

    // On envoie d'abord la taille de la string
    MPI_Bcast(&dir_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Les autres ranks préparent leur mémoire
    if (rank != 0) {
        params.run_dir.resize(dir_len);
    }

    // On envoie le contenu de la string (le buffer interne)
    if (dir_len > 0) {
        MPI_Bcast(&params.run_dir[0], dir_len, MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    fs::path base_path = fs::path(params.run_dir) / params.run_name;
    fs::path config_file = base_path / params.run_name;
    fs::path seed_file = base_path / (params.run_name + "_seed") /
                         (params.run_name + "_seed" + std::to_string(rank) + ".txt");

    bool local_existing = fs::exists(config_file) && fs::exists(seed_file);

    bool global_existing;
    MPI_Allreduce(&local_existing, &global_existing, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
    if (global_existing) {
        if (rank == 0) std::cout << "All ranks found existing configuration. Resuming...\n";
        return true;
    } else {
        return false;
    }
}

void print_parameters(const RunParamsECB& rp, const mpi::MpiTopology& topo) {
    if (topo.rank == 0) {
        std::cout << "==========================================" << std::endl;
        std::cout << "ECMC - MPI No Checkboard" << std::endl;
        std::cout << "==========================================" << std::endl;
        std::cout << "Run name : " + rp.run_name << "\n";
        std::cout << "---Lattice params---\n";
        std::cout << "Total lattice size : " << rp.L_core * rp.n_core_dims << "^4\n";
        std::cout << "Local lattice size : " << rp.L_core << "^4\n\n";

        std::cout << "---Run params---\n";
        std::cout << "Initial seed : " << rp.seed << "\n";
        std::cout << "Thermalization shifts : " << rp.N_therm << "\n";
        std::cout << "Number of shifts : " << rp.N_shift << "\n";
        std::cout << "Save each : " << rp.save_each_shifts << " shifts\n\n";

        std::cout << "---ECMC params---\n";
        std::cout << "Beta : " << rp.ecmc_params.beta << "\n";
        std::cout << "Theta sample : " << rp.ecmc_params.param_theta_sample << "\n";
        std::cout << "Theta refresh site : " << rp.ecmc_params.param_theta_refresh_site << "\n";
        std::cout << "Theta refresh R : " << rp.ecmc_params.param_theta_refresh_R << "\n";
        std::cout << "Epsilon set : " << rp.ecmc_params.epsilon_set << "\n";
        std::cout << "Poisson law : " << (rp.ecmc_params.poisson ? "Yes" : "No") << "\n\n";

        std::cout << "---Measures params---\n";
        std::cout << "Number of <P> samples : " << rp.N_shift / rp.N_shift_plaquette << "\n";
        std::cout << "Measure <P> each : " << rp.N_shift_plaquette << " shifts\n";
        std::cout << "Measure topo : " << (rp.topo ? "Yes" : "No") << "\n";
        if (rp.topo) {
            std::cout << "Number of Q samples : " << rp.N_shift / rp.N_shift_topo << "\n";
            std::cout << "Measure Q each : " << rp.N_shift_topo << " shifts\n\n";
            std::cout << "---Gradient Flow---\n";
            std::cout << "Number of RK3 steps per GF step : " << rp.N_rk_steps << "\n";
            std::cout << "Number of GF steps : " << rp.N_steps_gf << "\n";
        }
        std::cout << "==========================================" << std::endl;
    }
}

void io::save_params(const RunParamsECB& rp, const std::string& filename,
                     const std::string& dirpath) {
    // Create a data folder if doesn't exists
    fs::path base_dir(dirpath);
    fs::path dir = base_dir / filename;

    try {
        if (!fs::exists(dir)) {
            fs::create_directories(dir);
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Couldn't create folder data : " << e.what() << std::endl;
        return;
    }
    fs::path filepath = dir / (filename + "_params.txt");

    std::ofstream file(filepath, std::ios::out | std::ios::app);
    if (!file.is_open()) {
        std::cout << "Could not open file " << filepath << "\n";
        return;
    }
    file << std::boolalpha;

    file << "\n#########################################################\n\n";

    file << "# Lattice params\n";
    file << "L_core = " << rp.L_core << "\n";
    file << "n_core_dims = " << rp.n_core_dims << "\n";
    file << "cold_start = " << rp.cold_start << "\n\n";

    file << "# Run params\n";
    file << "seed = " << rp.seed << "\n";
    file << "N_shift_therm = " << rp.N_therm << "\n";
    file << "N_shift = " << rp.N_shift << "\n";

    file << "# ECMC params\n";
    file << "beta = " << rp.ecmc_params.beta << "\n";
    file << "theta_sample = " << rp.ecmc_params.param_theta_sample << "\n";
    file << "theta_refresh_site = " << rp.ecmc_params.param_theta_refresh_site << "\n";
    file << "theta_refresh_R = " << rp.ecmc_params.param_theta_refresh_R << "\n";
    file << "epsilon_set = " << rp.ecmc_params.epsilon_set << "\n";
    file << "poisson = " << rp.ecmc_params.poisson << "\n\n";

    file << "# Plaquette params\n";
    file << "N_shift_plaquette = " << rp.N_shift_plaquette << "\n\n";

    file << "# Topo params\n";
    file << "topo = " << rp.topo << "\n";
    file << "N_shift_topo = " << rp.N_shift_topo << "\n";
    file << "N_steps_gf = " << rp.N_steps_gf << "\n";
    file << "N_rk_steps = " << rp.N_rk_steps << "\n\n";

    file << "#Save params\n";
    file << "save_each_shifts = " << rp.save_each_shifts << "\n\n";

    file.close();
    std::cout << "Parameters saved in " << filepath << "\n";
};

// Reads the parameters of input file into RunParams struct
bool io::read_params(RunParamsECB& params, int rank, const std::string& input) {
    if (rank == 0) {
        try {
            // On suppose que load_params est surchargée pour RunParamsECB
            io::load_params(input, params);
        } catch (const std::exception& e) {
            std::cerr << "Error reading input : " << e.what() << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // 1. Diffusion des paramètres de base (Lattice + Run + Topo)
    MPI_Bcast(&params.L_core, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.n_core_dims, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.cold_start, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.N_shift, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.N_therm, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.topo, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.N_shift_plaquette, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.N_shift_topo, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.N_steps_gf, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.N_rk_steps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.save_each_shifts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 2. Diffusion du bloc ECMCParams (ecmc_params)
    MPI_Bcast(&params.ecmc_params.beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.ecmc_params.N_samples, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.ecmc_params.param_theta_sample, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.ecmc_params.param_theta_refresh_site, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.ecmc_params.param_theta_refresh_R, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.ecmc_params.poisson, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&params.ecmc_params.epsilon_set, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 3. Diffusion de la std::string (run_name)
    int name_len;
    if (rank == 0) {
        name_len = static_cast<int>(params.run_name.size());
    }

    MPI_Bcast(&name_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        params.run_name.resize(name_len);
    }

    if (name_len > 0) {
        MPI_Bcast(&params.run_name[0], name_len, MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    // 3. Diffusion de la std::string (run_dir)
    int dir_len;
    if (rank == 0) {
        dir_len = static_cast<int>(params.run_dir.size());
    }

    // On envoie d'abord la taille de la string
    MPI_Bcast(&dir_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Les autres ranks préparent leur mémoire
    if (rank != 0) {
        params.run_dir.resize(dir_len);
    }

    // On envoie le contenu de la string (le buffer interne)
    if (dir_len > 0) {
        MPI_Bcast(&params.run_dir[0], dir_len, MPI_CHAR, 0, MPI_COMM_WORLD);
    }
    // 4. Vérification de l'existence des fichiers pour la reprise (Resume)
    // On utilise std::filesystem (fs) comme dans ton exemple
    fs::path base_path = fs::path(params.run_dir) / params.run_name;
    fs::path config_file = base_path / params.run_name;
    fs::path seed_file = base_path / (params.run_name + "_seed") /
                         (params.run_name + "_seed" + std::to_string(rank) + ".txt");

    bool local_existing = fs::exists(config_file) && fs::exists(seed_file);

    bool global_existing;
    MPI_Allreduce(&local_existing, &global_existing, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
    if (global_existing) {
        if (rank == 0) std::cout << "All ranks found existing configuration. Resuming...\n";
        return true;
    } else {
        return false;
    }
}

// Adds the log of saved shift to params
void io::add_shift(int shift, const std::string& filename, const std::string& dirpath) {
    fs::path base_dir(dirpath);
    fs::path dir = base_dir / filename;

    try {
        if (!fs::exists(dir)) {
            fs::create_directories(dir);
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Couldn't create folder data : " << e.what() << std::endl;
        return;
    }
    fs::path filepath = dir / (filename + "_params.txt");

    std::ofstream file(filepath, std::ios::out | std::ios::app);
    if (!file.is_open()) {
        std::cout << "Could not open file " << filepath << "\n";
        return;
    }

    file << "Saved shift " << shift << "\n";
    file.close();
};

// Add finished to params
void io::add_finished(const std::string& filename, const std::string& dirpath) {
    fs::path base_dir(dirpath);
    fs::path dir = base_dir / filename;

    try {
        if (!fs::exists(dir)) {
            fs::create_directories(dir);
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Couldn't create folder data : " << e.what() << std::endl;
        return;
    }
    fs::path filepath = dir / (filename + "_params.txt");

    std::ofstream file(filepath, std::ios::out | std::ios::app);
    if (!file.is_open()) {
        std::cout << "Could not open file " << filepath << "\n";
        return;
    }

    file << "Saved final state !\n";
    file.close();
};
