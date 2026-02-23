#include "GeometryCB.h"

#include <cstdint>
//Initializes the precomputed tables neighbors, frozen and list_staples
GeometryCB::GeometryCB(int L_) {
    L_int = L_;
    V_int = L_int * L_int * L_int * L_int;
    L_ext = L_int + 2;
    V_ext = L_ext * L_ext * L_ext * L_ext;
    neighbors.resize((V_ext) * 8, SIZE_MAX);
    for (int t = 0; t < L_ext; t++) {
        for (int z = 0; z < L_ext; z++) {
            for (int y = 0; y < L_ext; y++) {
                for (int x = 0; x < L_ext; x++) {
                    size_t site = index(x, y, z, t);
                    // Direction X (mu=0)
                    if (x + 1 < L_ext) neighbors[index_neigh(site, 0, up)] = index(x + 1, y, z, t);
                    if (x - 1 >= 0) neighbors[index_neigh(site, 0, down)] = index(x - 1, y, z, t);
                    // Direction Y (mu=1)
                    if (y + 1 < L_ext) neighbors[index_neigh(site, 1, up)] = index(x, y + 1, z, t);
                    if (y - 1 >= 0) neighbors[index_neigh(site, 1, down)] = index(x, y - 1, z, t);
                    // Direction Z (mu=2)
                    if (z + 1 < L_ext) neighbors[index_neigh(site, 2, up)] = index(x, y, z + 1, t);
                    if (z - 1 >= 0) neighbors[index_neigh(site, 2, down)] = index(x, y, z - 1, t);
                    // Direction T (mu=3)
                    if (t + 1 < L_ext) neighbors[index_neigh(site, 3, up)] = index(x, y, z, t + 1);
                    if (t - 1 >= 0) neighbors[index_neigh(site, 3, down)] = index(x, y, z, t - 1);
                }
            }
        }
    }

    frozen.resize(V_ext * 4, false);
    // Frozen links are those that step out of the lattice core or belong to halos
    for (int t = 0; t < L_ext; t++) {
        for (int z = 0; z < L_ext; z++) {
            for (int y = 0; y < L_ext; y++) {
                for (int x = 0; x < L_ext; x++) {
                    bool link_is_frozen = false;
                    if (x == 0 or y == 0 or z == 0 or t == 0) link_is_frozen = true;
                    if (x == L_ext - 1 or y == L_ext - 1 or z == L_ext - 1 or t == L_ext - 1)
                        link_is_frozen = true;
                    for (int mu = 0; mu < 4; mu++) {
                        if (x == 1 and mu != 0) link_is_frozen = true;
                        if (y == 1 and mu != 1) link_is_frozen = true;
                        if (z == 1 and mu != 2) link_is_frozen = true;
                        if (t == 1 and mu != 3) link_is_frozen = true;
                        size_t i = index(x, y, z, t);
                        frozen[index_frozen(i, mu)] = link_is_frozen;
                    }
                }
            }
        }
    }

    links_staples.resize(V_ext * 4 * 6 * 3, std::make_pair(SIZE_MAX, -1));
    // 4 links per site, 6 staples per link, 3 links per staple
    for (int t = 0; t < L_ext; t++) {
        for (int z = 0; z < L_ext; z++) {
            for (int y = 0; y < L_ext; y++) {
                for (int x = 0; x < L_ext; x++) {
                    size_t site = index(x, y, z, t);  // x
                    for (int mu = 0; mu < 4; mu++) {
                        if (!is_frozen(site, mu)) {
                            int j = 0;
                            for (int nu = 0; nu < 4; nu++) {
                                if (nu == mu) continue;

                                size_t xmu = get_neigh(site, mu, up);     // x+mu
                                size_t xnu = get_neigh(site, nu, up);     // x+nu
                                size_t xmunu = get_neigh(xmu, nu, down);  // x+mu-nu
                                size_t xmnu = get_neigh(site, nu, down);  // x-nu

                                links_staples[index_staples(site, mu, j, 0)] = {xmu, nu};
                                links_staples[index_staples(site, mu, j, 1)] = {xnu, mu};
                                links_staples[index_staples(site, mu, j, 2)] = {site, nu};
                                links_staples[index_staples(site, mu, j + 1, 0)] = {xmunu, nu};
                                links_staples[index_staples(site, mu, j + 1, 1)] = {xmnu, mu};
                                links_staples[index_staples(site, mu, j + 1, 2)] = {xmnu, nu};

                                j += 2;
                            }
                        }
                    }
                }
            }
        }
    }
}
