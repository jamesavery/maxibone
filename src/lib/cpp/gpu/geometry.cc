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

bool in_bbox(float U, float V, float W, const std::array<float, 6> &bbox) {
    return cpu_seq::in_bbox(U, V, W, bbox);
}

array<real_t,9> inertia_matrix(const input_ndarray<mask_type> &mask, const array<real_t,3> &cm) {
    return cpu_seq::inertia_matrix(mask, cm);
}

void integrate_axes(const input_ndarray<mask_type> &mask,
		    const array<real_t,3> &x0,
		    const array<real_t,3> &v_axis,
		    const array<real_t,3> &w_axis,
		    const real_t v_min, const real_t w_min,
		    output_ndarray<real_t> output) {
    return cpu_seq::integrate_axes(mask, x0, v_axis, w_axis, v_min, w_min, output);
}

template <typename T>
float resample2x2x2(const T        *voxels,
                    const array<ssize_t, 3> &shape,
                    const array<float, 3>   &X) {
    return cpu_seq::resample2x2x2(voxels, shape, X);
}

template <typename T>
void sample_plane(const input_ndarray<T> &voxels,
                  const real_t voxel_size, // In micrometers
                  const array<real_t, 3> cm,
                  const array<real_t, 3> u_axis,
                  const array<real_t, 3> v_axis,
                  const array<real_t, 4> bbox,    // [umin,umax,vmin,vmax] in micrometers
                  output_ndarray<real_t> plane_samples) {
    return cpu_seq::sample_plane(voxels, voxel_size, cm, u_axis, v_axis, bbox, plane_samples);
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