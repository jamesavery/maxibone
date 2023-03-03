#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <math.h>
using namespace std;

#include "geometry.hh"
#include "../cpu_seq/geometry.cc"

namespace cpu_par {

// TODO look at function aliasing. Currently doesn't work, as it clashes with the header file prototype.
array<real_t,3> center_of_mass(const input_ndarray<mask_type> &mask) {
    return cpu_seq::center_of_mass(mask);
}

bool in_bbox(float U, float V, float W, const std::array<float, 6> &bbox) {
    return cpu_seq::in_bbox(U, V, W, bbox);
}

array<real_t,9> inertia_matrix(const input_ndarray<mask_type> &mask, const array<real_t,3> &cm) {
    return cpu_seq::inertia_matrix(mask, cm);
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

}