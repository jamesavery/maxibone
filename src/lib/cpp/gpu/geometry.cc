#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <math.h>
using namespace std;

#include "geometry.hh"
#include "../cpu_seq/geometry.cc"

namespace gpu {

array<real_t,3> center_of_mass(const input_ndarray<mask_type> &mask) {
    return cpu_seq::center_of_mass(mask);
}

array<real_t,9> inertia_matrix(const input_ndarray<mask_type> &voxels, const array<real_t,3> &cm) {
    // nvc++ doesn't support OpenACC 2.7 array reductions yet, so must name each element.
    real_t
        Ixx = 0, Ixy = 0, Ixz = 0,
                 Iyy = 0, Iyz = 0,
                          Izz = 0;

    size_t Nx = voxels.shape[0], Ny = voxels.shape[1], Nz = voxels.shape[2];
    ssize_t image_length = Nx*Ny*Nz;

    print_timestamp("inertia_matrix start");

    #pragma acc data copy(Ixx, Ixy, Ixz, Iyy, Iyz, Izz)
    {
        for (ssize_t block_start = 0; block_start < image_length; block_start += acc_block_size<mask_type>) {
            const mask_type *buffer = voxels.data + block_start;
            ssize_t this_block_size = min(acc_block_size<mask_type>, image_length - block_start);

            #pragma acc data copyin(buffer[:this_block_size])
            {
                #pragma acc parallel loop reduction(+:Ixx,Iyy,Izz) reduction(-:Ixy,Ixz,Iyz)
                for (int64_t k = 0; k < this_block_size; k++) {    //\if (buffer[k] != 0)
                    mask_type m = buffer[k];

                    // m guards this, and GPUs doesn't like branches
                    //if (m != 0)
                    int64_t
                        flat_idx = block_start + k,
                        X = flat_idx / (Ny * Nz),
                        Y = ((flat_idx) / Nz) % Ny,
                        Z = flat_idx % Nz;

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
    }

    print_timestamp("inertia_matrix end");

    return array<real_t,9> {
        Ixx, Ixy, Ixz,
        Ixy, Iyy, Iyz,
        Ixz, Iyz, Izz
    };
}

/* TODO Only called in test.py. Postponed for now.
void integrate_axes(const input_ndarray<mask_type> &voxels,
            const array<real_t,3> &x0,
            const array<real_t,3> &v_axis,
            const array<real_t,3> &w_axis,
            const real_t v_min, const real_t w_min,
            output_ndarray<real_t> output) {
    ssize_t Nx = voxels.shape[0], Ny = voxels.shape[1], Nz = voxels.shape[2];
    ssize_t Nv = output.shape[0], Nw = output.shape[1];
    int64_t image_length = Nx*Ny*Nz;
    real_t *output_data = output.data;

    // TODO: Check v_axis & w_axis projections to certify bounds and get rid of runtime check

    for (ssize_t block_start = 0; block_start < image_length; block_start += acc_block_size) {
        const mask_type *buffer  = voxels.data + block_start;
        int block_length = min(acc_block_size,image_length-block_start);

        //#pragma acc parallel loop copy(output_data[:Nv*Nw]) copyin(buffer[:block_length], x0, v_axis, w_axis)
        //parallel_loop((output_data[:Nv*Nw]))
        for (int64_t k = 0; k < block_length; k++) {
            if (buffer[k] != 0) {
                int64_t flat_idx = block_start + k;
                real_t xs[3] = {
                    (flat_idx  / (Ny*Nz))  - x0[0],   // x
                    ((flat_idx / Nz) % Ny) - x0[1],   // y
                    (flat_idx  % Nz)       - x0[2] }; // z

                mask_type voxel = buffer[k];
                real_t v = dot(xs,v_axis), w = dot(xs,w_axis);
                int64_t i_v = round(v-v_min), j_w = round(w-w_min);

                if (i_v >= 0 && j_w >= 0 && i_v < Nv && j_w < Nw) {
                    //atomic_statement()
                    output_data[i_v*Nw + j_w] += voxel;
                }
            }
        }
    }
}
*/

}