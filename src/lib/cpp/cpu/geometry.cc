#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <math.h>
using namespace std;

#include "geometry.hh"
#include "../cpu_seq/geometry.cc"

namespace cpu_par {

array<real_t,3> center_of_mass(const input_ndarray<mask_type> &mask) {
    return cpu_seq::center_of_mass(mask);
}

array<real_t,9> inertia_matrix(const input_ndarray<mask_type> &voxels, const array<real_t,3> &cm) {
    real_t
        Ixx = 0, Ixy = 0, Ixz = 0,
                 Iyy = 0, Iyz = 0,
                          Izz = 0;

    ssize_t Nx = voxels.shape[0], Ny = voxels.shape[1], Nz = voxels.shape[2];

    print_timestamp("inertia_matrix_serial start");

    #pragma omp parallel for collapse(3) reduction(+:Ixx,Iyy,Izz) reduction(-:Ixy,Ixz,Iyz)
    for (int64_t X = 0; X < Nx; X++) {
        for (int64_t Y = 0; Y < Ny; Y++) {
            for (int64_t Z = 0; Z < Nz; Z++) {

                // TODO shouldn't the loops be interchanged to match the access pattern? (Naming-wise, that is)
                int64_t k = X*Ny*Nz + Y*Nz + Z;
                mask_type m = voxels.data[k];

                // m guards this, and this removes branches
                // if (m != 0)
                real_t
                    x = X - cm[0],
                    y = Y - cm[1],
                    z = Z - cm[2];

                Ixx += m * (y*y + z*z);
                Iyy += m * (x*x + z*z);
                Izz += m * (x*x + y*y);
                Ixy -= m * x*y;
                Ixz -= m * x*z;
                Iyz -= m * y*z;
            }
        }
    }

    print_timestamp("inertia_matrix_serial end");

    return array<real_t,9> {
        Ixx, Ixy, Ixz,
        Ixy, Iyy, Iyz,
        Ixz, Iyz, Izz
    };
}

}