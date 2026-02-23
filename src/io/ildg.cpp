#include "ildg.h"

#include <filesystem>
#include <iostream>

#include "../mpi/HalosExchange.h"
extern "C" {
#include <lime.h>
}

namespace fs = std::filesystem;

std::string generate_ildg_xml(int L_glob) {
    char buf[512];
    std::sprintf(buf,
                 "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                 "<ildgFormat><version>1.0</version><field>su3gauge</field>"
                 "<precision>64</precision><lx>%d</lx><ly>%d</ly><lz>%d</lz><lt>%d</lt>"
                 "</ildgFormat>",
                 L_glob, L_glob, L_glob, L_glob);
    return std::string(buf);
}

// Conversion en Big-Endian (requis par la norme ILDG)
inline double swap_double_endian(double val) {
    uint64_t bits;
    std::memcpy(&bits, &val, sizeof(bits));
    bits = __builtin_bswap64(bits);
    double swapped;
    std::memcpy(&swapped, &bits, sizeof(swapped));
    return swapped;
}

void save_ildg_clime(const std::string& filename, const std::string& dirpath,
                     const GaugeField& field, const GeometryCB& geo, const mpi::MpiTopology& topo) {
    int L_global = geo.L_int * topo.n_core_dim;
    long nbytes_binary = (long)L_global * L_global * L_global * L_global * 72 * sizeof(double);
    MPI_Offset header_offset = 0;

    fs::path dir(dirpath);
    fs::path file = dir / filename /filename;
    // --- ÉTAPE 1 : Le Rang 0 écrit les Headers LIME ---
    if (topo.rank == 0) {
        if (!fs::exists(dir)) {
            fs::create_directories(dir);
        }
        FILE* fp = fopen(file.string().c_str(), "w");
        if (!fp) {
            std::pair<std::string, int> err_info(file.string(), errno);
            std::cerr << "Erreur critique : Impossible d'ouvrir le fichier en écriture : "
                      << err_info.first << " (errno: " << err_info.second << ")" << std::endl;
            header_offset = -1;  // Signal d'erreur pour le Bcast
        } else {
            LimeWriter* w = limeCreateWriter(fp);

            // 1. Record XML (ildg-format)
            std::string xml = generate_ildg_xml(L_global);
            n_uint64_t xml_len = xml.size();
            LimeRecordHeader* h_xml = limeCreateHeader(1, 1, (char*)"ildg-format", xml.size());
            limeWriteRecordHeader(h_xml, w);
            limeWriteRecordData((void*)xml.c_str(), &xml_len, w);
            limeDestroyHeader(h_xml);

            // 2. Record Binaire (ildg-binary-data)
            // On écrit juste le header pour réserver l'espace des données
            LimeRecordHeader* h_bin =
                limeCreateHeader(0, 1, (char*)"ildg-binary-data", nbytes_binary);
            limeWriteRecordHeader(h_bin, w);

            // On récupère la position actuelle pour dire aux autres où commencer à écrire
            header_offset = (MPI_Offset)ftell(fp);

            limeDestroyWriter(w);
            fclose(fp);
            if (topo.rank == 0) {
                std::cout << "Configuration saved in " << file << "\n";
            }
        }
    }

    MPI_Barrier(topo.cart_comm);
    // --- ÉTAPE 2 : Synchronisation de l'offset ---
    MPI_Bcast(&header_offset, 1, MPI_OFFSET, 0, topo.cart_comm);
    // --- ÉTAPE 3 : Écriture MPI-I/O (Identique à la fonction précédente) ---
    // On réutilise la logique de local_buffer et de subarray

    std::vector<double> local_buffer(geo.V_int * 72);
    size_t buf_idx = 0;
    for (int t = 1; t <= geo.L_int; ++t) {
        for (int z = 1; z <= geo.L_int; ++z) {
            for (int y = 1; y <= geo.L_int; ++y) {
                for (int x = 1; x <= geo.L_int; ++x) {
                    size_t site = geo.index(x, y, z, t);
                    for (int mu = 0; mu < 4; ++mu) {
                        auto link = field.view_link_const(site, mu);
                        for (int i = 0; i < 3; ++i) {
                            for (int j = 0; j < 3; ++j) {
                                local_buffer[buf_idx++] = swap_double_endian(link(i, j).real());
                                local_buffer[buf_idx++] = swap_double_endian(link(i, j).imag());
                            }
                        }
                    }
                }
            }
        }
    }

    MPI_File fh;
    int err =
        MPI_File_open(topo.cart_comm, file.string().c_str(), MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    if (err != MPI_SUCCESS) {
        if (topo.rank == 0)
            std::cerr << "Fatal: Cannot open MPI file for writing: " << file << std::endl;
        MPI_Abort(topo.cart_comm, err);
    }
    // Définition du type de donnée et du subarray (comme avant)
    MPI_Datatype site_type, file_type;
    MPI_Type_contiguous(72, MPI_DOUBLE, &site_type);
    MPI_Type_commit(&site_type);

    int g_sz[4] = {L_global, L_global, L_global, L_global};
    int l_sz[4] = {geo.L_int, geo.L_int, geo.L_int, geo.L_int};
    int start[4] = {topo.local_coords[3] * geo.L_int, topo.local_coords[2] * geo.L_int,
                    topo.local_coords[1] * geo.L_int, topo.local_coords[0] * geo.L_int};

    MPI_Type_create_subarray(4, g_sz, l_sz, start, MPI_ORDER_C, site_type, &file_type);
    MPI_Type_commit(&file_type);

    // APPLICATION DE L'OFFSET LIME
    MPI_File_set_view(fh, (MPI_Offset)header_offset, site_type, file_type, "native", MPI_INFO_NULL);
    MPI_File_write_all(fh, local_buffer.data(), (int)geo.V_int, site_type, MPI_STATUS_IGNORE);

    MPI_File_close(&fh);
    MPI_Type_free(&file_type);
    MPI_Type_free(&site_type);
}

void read_ildg_clime(const std::string& filename, const std::string& dirpath, GaugeField& field,
                     const GeometryCB& geo, mpi::MpiTopology& topo) {
    MPI_Offset header_offset = 0;
    int L_global = geo.L_int * topo.n_core_dim;
    MPI_Offset expected_bytes =
        static_cast<MPI_Offset>(L_global) * L_global * L_global * L_global * 72 * sizeof(double);

    fs::path dir(dirpath);
    fs::path file = dir / filename /filename;
    // --- ÉTAPE 1 : Le Rang 0 cherche le record binaire ---
    if (topo.rank == 0) {
        FILE* fp = fopen(file.string().c_str(), "r");
        if (!fp) {
            std::cerr << "Erreur : Impossible d'ouvrir " << file << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        LimeReader* r = limeCreateReader(fp);
        int status;

        // On parcourt les records un par un
        while ((status = limeReaderNextRecord(r)) != LIME_EOF) {
            char* type = limeReaderType(r);
            n_uint64_t nbytes = limeReaderBytes(r);

            if (std::string(type) == "ildg-binary-data") {
                if ((MPI_Offset)nbytes != expected_bytes) {
                    std::cerr << "Erreur : Taille binaire incorrecte dans le fichier !"
                              << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                // L'offset est la position actuelle du pointeur de fichier
                header_offset = ftello(fp);
                break;
            }
        }
        limeDestroyReader(r);
        fclose(fp);
    }

    // --- ÉTAPE 2 : Diffusion de l'offset à tous les rangs ---
    MPI_Bcast(&header_offset, 1, MPI_OFFSET, 0, topo.cart_comm);

    if (header_offset == 0) {
        if (topo.rank == 0)
            std::cerr << "Erreur : Record 'ildg-binary-data' non trouvé." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // --- ÉTAPE 3 : Lecture Parallèle avec MPI-I/O ---
    std::vector<double> local_buffer(geo.V_int * 72);

    MPI_File fh;
    MPI_File_open(topo.cart_comm, file.string().c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    // Configuration de la vue (même logique que pour save_ildg)
    MPI_Datatype site_type, file_type;
    MPI_Type_contiguous(72, MPI_DOUBLE, &site_type);
    MPI_Type_commit(&site_type);

    int g_sz[4] = {L_global, L_global, L_global, L_global};
    int l_sz[4] = {geo.L_int, geo.L_int, geo.L_int, geo.L_int};
    int start[4] = {topo.local_coords[3] * geo.L_int, topo.local_coords[2] * geo.L_int,
                    topo.local_coords[1] * geo.L_int, topo.local_coords[0] * geo.L_int};

    MPI_Type_create_subarray(4, g_sz, l_sz, start, MPI_ORDER_C, site_type, &file_type);
    MPI_Type_commit(&file_type);

    MPI_File_set_view(fh, header_offset, site_type, file_type, "native", MPI_INFO_NULL);
    MPI_File_read_all(fh, local_buffer.data(), static_cast<int>(geo.V_int), site_type,
                      MPI_STATUS_IGNORE);

    MPI_File_close(&fh);

    // --- ÉTAPE 4 : Répartition dans GaugeField et Byte-Swap ---
    size_t buf_idx = 0;
    for (int t = 1; t <= geo.L_int; ++t) {
        for (int z = 1; z <= geo.L_int; ++z) {
            for (int y = 1; y <= geo.L_int; ++y) {
                for (int x = 1; x <= geo.L_int; ++x) {
                    size_t site = geo.index(x, y, z, t);

                    for (int mu = 0; mu < 4; ++mu) {
                        for (int i = 0; i < 3; ++i) {
                            for (int j = 0; j < 3; ++j) {
                                double re = swap_double_endian(local_buffer[buf_idx++]);
                                double im = swap_double_endian(local_buffer[buf_idx++]);
                                field.view_link(site, mu)(i, j) = Complex(re, im);
                            }
                        }
                    }
                }
            }
        }
    }

    MPI_Type_free(&file_type);
    MPI_Type_free(&site_type);
    mpi::exchange::exchange_halos_cascade(field, geo, topo);
}
