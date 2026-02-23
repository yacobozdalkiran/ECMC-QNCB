#include "Shift.h"
#include <iostream>

#include "HalosExchange.h"

//Packing into buffers
void mpi::shift::pack_shift_slice(const GaugeField& field, const GeometryCB& geo,
                                  std::vector<Complex>& buffer, int dim, int start, int end) {
    size_t idx_buf = 0;
    int c[4];

    // Bornes par défaut : tout le volume intérieur
    int min_c[4] = {1, 1, 1, 1};
    int max_c[4] = {geo.L_int, geo.L_int, geo.L_int, geo.L_int};

    // On restreint la dimension de transfert à la tranche [start, end]
    min_c[dim] = start;
    max_c[dim] = end;

    for (c[3] = min_c[3]; c[3] <= max_c[3]; ++c[3]) {
        for (c[2] = min_c[2]; c[2] <= max_c[2]; ++c[2]) {
            for (c[1] = min_c[1]; c[1] <= max_c[1]; ++c[1]) {
                for (c[0] = min_c[0]; c[0] <= max_c[0]; ++c[0]) {
                    size_t site_idx = geo.index(c[0], c[1], c[2], c[3]);

                    for (int mu = 0; mu < 4; ++mu) {
                        // Copie optimisée via Eigen
                        Eigen::Map<SU3> buffer_map(&buffer[idx_buf]);
                        buffer_map = field.view_link_const(site_idx, mu);
                        idx_buf += 9;
                    }
                }
            }
        }
    }
}

//Unpacking into the field
void mpi::shift::unpack_shift_slice(GaugeField& field, const GeometryCB& geo,
                                    const std::vector<Complex>& buffer, int dim, int start,
                                    int end) {
    size_t idx_buf = 0;
    int c[4];
    int min_c[4] = {1, 1, 1, 1};
    int max_c[4] = {geo.L_int, geo.L_int, geo.L_int, geo.L_int};

    // La destination est toujours le début (1 à s) ou la fin (L-s+1 à L)
    min_c[dim] = start;
    max_c[dim] = end;

    for (c[3] = min_c[3]; c[3] <= max_c[3]; ++c[3]) {
        for (c[2] = min_c[2]; c[2] <= max_c[2]; ++c[2]) {
            for (c[1] = min_c[1]; c[1] <= max_c[1]; ++c[1]) {
                for (c[0] = min_c[0]; c[0] <= max_c[0]; ++c[0]) {
                    size_t site_idx = geo.index(c[0], c[1], c[2], c[3]);

                    for (int mu = 0; mu < 4; ++mu) {
                        // Écriture optimisée via Eigen
                        field.view_link(site_idx, mu) = Eigen::Map<const SU3>(&buffer[idx_buf]);
                        idx_buf += 9;
                    }
                }
            }
        }
    }
}

//Local shift of the field
void mpi::shift::apply_local_shift(GaugeField& field, const GeometryCB& geo, int dim, int s) {
    if (s == 0) return;  // Rien à faire

    int L = geo.L_int;

    // Pour un shift de s > 0 (vers la droite/haut), on doit
    // parcourir la dimension 'dim' à l'ENVERS pour ne pas écraser
    // les données dont on aura besoin plus tard.

    // On définit les bornes de la boucle pour la dimension du shift
    int start = L;
    int end = s + 1;
    int step = -1;

    // On parcourt la direction du shift à l'envers
    for (int t = (dim == 3 ? start : 1); (dim == 3 ? t != end + step : t <= L);
         t += (dim == 3 ? step : 1)) {
        for (int z = (dim == 2 ? start : 1); (dim == 2 ? z != end + step : z <= L);
             z += (dim == 2 ? step : 1)) {
            for (int y = (dim == 1 ? start : 1); (dim == 1 ? y != end + step : y <= L);
                 y += (dim == 1 ? step : 1)) {
                for (int x = (dim == 0 ? start : 1); (dim == 0 ? x != end + step : x <= L);
                     x += (dim == 0 ? step : 1)) {
                    int c_dst[4] = {x, y, z, t};
                    int c_src[4] = {x, y, z, t};
                    c_src[dim] = c_dst[dim] - s;

                    size_t src_idx = geo.index(c_src[0], c_src[1], c_src[2], c_src[3]);
                    size_t dst_idx = geo.index(c_dst[0], c_dst[1], c_dst[2], c_dst[3]);

                    for (int mu = 0; mu < 4; ++mu) {
                        field.view_link(dst_idx, mu) = field.view_link_const(src_idx, mu);
                    }
                }
            }
        }
    }
}

// Shifts the content of the lattices of s
void mpi::shift::shift_field(GaugeField& field, const GeometryCB& geo, HalosShift& h,
                             mpi::MpiTopology& topo, const int s[4]) {
    for (int dim = 0; dim < 4; ++dim) {
        if (s[dim] == 0) continue;  // Pas de décalage pour cette dimension

        int shift = s[dim];
        // 1. Calcul de la taille du buffer (volume de la tranche à envoyer)
        // La tranche à envoyer a une épaisseur de 'shift' sites
        size_t slice_volume = shift * geo.L_int * geo.L_int * geo.L_int;
        size_t buffer_size = slice_volume * 4 * 9;

        // 2. PACKING : On prépare les données qui SORTENT du processeur
        // Pour un shift positif, ce sont les 'shift' dernières tranches
        // Coordonnées de la tranche à envoyer : [L_int - shift + 1, L_int]
        pack_shift_slice(field, geo, h.send, dim, geo.L_int - shift + 1, geo.L_int);

        // 3. MPI SENDRECV
        int rank_up, rank_down;
        // On récupère les voisins selon la dimension (topo.xL, topo.x0, etc.)
        switch (dim) {
            case 0:
                rank_up = topo.xL;
                rank_down = topo.x0;
                break;
            case 1:
                rank_up = topo.yL;
                rank_down = topo.y0;
                break;
            case 2:
                rank_up = topo.zL;
                rank_down = topo.z0;
                break;
            case 3:
                rank_up = topo.tL;
                rank_down = topo.t0;
                break;
        }

        // On envoie vers le haut, on reçoit du bas
        MPI_Sendrecv(h.send.data(), buffer_size, MPI_DOUBLE_COMPLEX, rank_up, 0, h.recv.data(),
                     buffer_size, MPI_DOUBLE_COMPLEX, rank_down, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

        // 4. SHIFT LOCAL : On décale les données qui restent dans le processeur
        // On le fait avant d'UNPACK pour ne pas écraser les données sources
        apply_local_shift(field, geo, dim, shift);

        // 5. UNPACKING : On place les données reçues au début de la lattice locale
        // Coordonnées de destination : [1, shift]
        unpack_shift_slice(field, geo, h.recv, dim, 1, shift);
    }
}

//Performs a random shift
void mpi::shift::random_shift(GaugeField& field, const GeometryCB& geo, HalosShift& h,
                              mpi::MpiTopology& topo, std::mt19937_64& rng) {
    int shift_x{}, shift_y{}, shift_z{}, shift_t{};
    std::uniform_int_distribution<int> rand_l_shift(0, geo.L_int - 1);
    shift_x = rand_l_shift(rng);
    shift_y = rand_l_shift(rng);
    shift_z = rand_l_shift(rng);
    shift_t = rand_l_shift(rng);

    MPI_Bcast(&shift_x, 1, MPI_INT, 0, topo.cart_comm);
    MPI_Bcast(&shift_y, 1, MPI_INT, 0, topo.cart_comm);
    MPI_Bcast(&shift_z, 1, MPI_INT, 0, topo.cart_comm);
    MPI_Bcast(&shift_t, 1, MPI_INT, 0, topo.cart_comm);

    if (topo.rank == 0) {
        std::cout << "Shift (" << shift_x << ", " << shift_y << ", " << shift_z << ", " << shift_t
                  << ")\n";
    }
    int s[4] = {shift_x, shift_y, shift_z, shift_t};
    shift_field(field, geo, h, topo, s);
    // Après un shift global, les halos sont obsolètes
    // Il faut relancer l'échange de halos.
    mpi::exchange::exchange_halos_cascade(field, geo, topo);
};
