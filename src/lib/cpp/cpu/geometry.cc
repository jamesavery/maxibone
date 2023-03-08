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

void compute_front_mask(const input_ndarray<mask_type> solid_implant,
        const float voxel_size,
        const matrix4x4 &Muvw,
        std::array<float,6> bbox,
        output_ndarray<mask_type> front_mask) {
    return cpu_seq::compute_front_mask(solid_implant, voxel_size, Muvw, bbox, front_mask);
}

void fill_implant_mask(const input_ndarray<mask_type> mask,
               float voxel_size,
               const array<float,6> &bbox,
               float r_fraction,
               const matrix4x4 &Muvw,
               output_ndarray<mask_type> solid_implant_mask,
               output_ndarray<float> rsqr_maxs,
               output_ndarray<float> profile) {
    return cpu_seq::fill_implant_mask(mask, voxel_size, bbox, r_fraction, Muvw, solid_implant_mask, rsqr_maxs, profile);
}

array<real_t,9> inertia_matrix(const input_ndarray<mask_type> &mask, const array<real_t,3> &cm) {
    return cpu_seq::inertia_matrix(mask, cm);
}

void integrate_axes(const input_ndarray<mask_type> &mask,
		    const array<real_t,3> &x0,
		    const array<real_t,3> &v_axis,
		    const array<real_t,3> &w_axis,
		    const real_t v_min, const real_t w_min,
		    output_ndarray<uint64_t> output) {
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

void zero_outside_bbox(const array<real_t,9> &principal_axes,
                       const array<real_t,6> &parameter_ranges,
                       const array<real_t,3> &cm,
                       output_ndarray<mask_type> voxels) {
    return cpu_seq::zero_outside_bbox(principal_axes, parameter_ranges, cm, voxels);
}

}