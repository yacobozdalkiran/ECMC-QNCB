#include "HalosExchange.h"

//Exchanges the halos between cores in cascade (for corners)
void mpi::exchange::exchange_halos_cascade(GaugeField& field, const GeometryCB& geo,
                                           mpi::MpiTopology& topo) {
    int min_coord[4] = {1, 1, 1, 1};
    int max_coord[4] = {geo.L_int, geo.L_int, geo.L_int, geo.L_int};

    for (int dim = 0; dim < 4; ++dim) {
        // 1. Calculer la taille du buffer pour cette étape
        size_t face_volume = 1;
        for (int d = 0; d < 4; ++d) {
            if (d != dim) face_volume *= (max_coord[d] - min_coord[d] + 1);
        }
        // Chaque site a 4 liens de 9 complexes
        size_t buffer_size = face_volume * 4 * 9;

        std::vector<Complex> send_up(buffer_size), send_down(buffer_size);
        std::vector<Complex> recv_up(buffer_size), recv_down(buffer_size);

        // 2. PACKING (Remplissage des buffers d'envoi)
        pack_face(field, geo, send_up, dim, geo.L_int, min_coord,
                  max_coord);  // On envoie la face L vers le haut
        pack_face(field, geo, send_down, dim, 1, min_coord,
                  max_coord);  // On envoie la face 1 vers le bas

        // 3. COMMUNICATIONS MPI (Non-bloquantes)
        MPI_Request reqs[4];
        int rank_up, rank_down;
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

        MPI_Isend(send_up.data(), buffer_size, MPI_DOUBLE_COMPLEX, rank_up, 0, MPI_COMM_WORLD,
                  &reqs[0]);
        MPI_Isend(send_down.data(), buffer_size, MPI_DOUBLE_COMPLEX, rank_down, 1, MPI_COMM_WORLD,
                  &reqs[1]);

        MPI_Irecv(recv_down.data(), buffer_size, MPI_DOUBLE_COMPLEX, rank_down, 0, MPI_COMM_WORLD,
                  &reqs[2]);
        MPI_Irecv(recv_up.data(), buffer_size, MPI_DOUBLE_COMPLEX, rank_up, 1, MPI_COMM_WORLD,
                  &reqs[3]);

        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        // 4. UNPACKING (Écriture dans les halos de la lattice)
        unpack_face(field, geo, recv_up, dim, geo.L_int + 1, min_coord,
                    max_coord);  // Le voisin du haut écrit dans notre halo L+1
        unpack_face(field, geo, recv_down, dim, 0, min_coord,
                    max_coord);  // Le voisin du bas écrit dans notre halo 0

        // 5. EXTENSION DE LA BOÎTE (Magie de la cascade)
        // Les prochains transferts incluront les halos de cette dimension !
        min_coord[dim] = 0;
        max_coord[dim] = geo.L_int + 1;
    }
};

//Face packing into buffers
void mpi::exchange::pack_face(const GaugeField& field, const GeometryCB& geo,
                              std::vector<Complex>& buffer, int dim, int face_coord,
                              const int min_c[4], const int max_c[4]) {
    size_t idx_buf = 0;
    int c[4];  // Coordonnées actuelles [x, y, z, t]

    // On fixe la coordonnée de la dimension de coupe
    c[dim] = face_coord;

    // Boucles sur les autres dimensions
    // L'ordre t, z, y, x est respecté pour optimiser le cache mémoire
    for (c[3] = (dim == 3 ? face_coord : min_c[3]); c[3] <= (dim == 3 ? face_coord : max_c[3]);
         ++c[3]) {
        for (c[2] = (dim == 2 ? face_coord : min_c[2]); c[2] <= (dim == 2 ? face_coord : max_c[2]);
             ++c[2]) {
            for (c[1] = (dim == 1 ? face_coord : min_c[1]);
                 c[1] <= (dim == 1 ? face_coord : max_c[1]); ++c[1]) {
                for (c[0] = (dim == 0 ? face_coord : min_c[0]);
                     c[0] <= (dim == 0 ? face_coord : max_c[0]); ++c[0]) {
                    // Si on est sur l'axe de transfert, on ne boucle pas vraiment dessus
                    if (dim == 0 && c[0] != face_coord) continue;
                    if (dim == 1 && c[1] != face_coord) continue;
                    if (dim == 2 && c[2] != face_coord) continue;
                    if (dim == 3 && c[3] != face_coord) continue;

                    // Récupération de l'index 1D dans le tableau (L+2)^4
                    size_t site_idx = geo.index(c[0], c[1], c[2], c[3]);

                    // Copie des 4 liens vers le buffer
                    for (int mu = 0; mu < 4; ++mu) {
                        Eigen::Map<const SU3> link_map = field.view_link_const(site_idx, mu);
                        Eigen::Map<SU3> buffer_map(&buffer[idx_buf]);

                        buffer_map = link_map;  // Copie la matrice 3x3 d'un coup
                        idx_buf += 9;
                    }
                }
            }
        }
    }
};

//Face unpacking into field
void mpi::exchange::unpack_face(GaugeField& field, const GeometryCB& geo,
                                const std::vector<Complex>& buffer, int dim, int face_coord,
                                const int min_c[4], const int max_c[4]) {
    size_t idx_buf = 0;
    int c[4];  // Tableau pour stocker [x, y, z, t] en cours

    // On parcourt de T (dim 3) vers X (dim 0)
    // L'opérateur ternaire fixe la coordonnée si c'est la dimension de transfert,
    // sinon il boucle de min_c à max_c.
    for (c[3] = (dim == 3 ? face_coord : min_c[3]); c[3] <= (dim == 3 ? face_coord : max_c[3]);
         ++c[3]) {
        for (c[2] = (dim == 2 ? face_coord : min_c[2]); c[2] <= (dim == 2 ? face_coord : max_c[2]);
             ++c[2]) {
            for (c[1] = (dim == 1 ? face_coord : min_c[1]);
                 c[1] <= (dim == 1 ? face_coord : max_c[1]); ++c[1]) {
                for (c[0] = (dim == 0 ? face_coord : min_c[0]);
                     c[0] <= (dim == 0 ? face_coord : max_c[0]); ++c[0]) {
                    // 1. Récupération de l'index 1D dans le tableau (L+2)^4
                    // On suppose ici que get_index est accessible (soit passée, soit méthode de
                    // field)
                    size_t site_idx = geo.index(c[0], c[1], c[2], c[3]);

                    // 2. Déchargement du buffer vers les 4 liens du site
                    for (int mu = 0; mu < 4; ++mu) {
                        Eigen::Map<SU3> link_map = field.view_link(site_idx, mu);
                        Eigen::Map<const SU3> buffer_map(&buffer[idx_buf]);

                        link_map = buffer_map;  // Copie la matrice 3x3 d'un coup
                        idx_buf += 9;
                    }
                }
            }
        }
    }
}
