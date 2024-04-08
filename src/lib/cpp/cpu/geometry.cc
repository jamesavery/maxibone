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

void cylinder_projection(const input_ndarray<float>  edt,  // Euclidean Distance Transform in um, should be low-resolution (will be interpolated)
             const input_ndarray<uint8_t> C,  // Material classification images (probability per voxel, 0..1 -> 0..255)
             float voxel_size,           // Voxel size for Cs
             float d_min, float d_max,       // Distance shell to map to cylinder
             float theta_min, float theta_max, // Angle range (wrt cylinder center)
             std::array<float,6> bbox,
             const matrix4x4 &Muvw,           // Transform from zyx (in um) to U'V'W' cylinder FoR (in um)
             output_ndarray<float>    image,  // Probability-weighted volume of (class,theta,U)-voxels
             output_ndarray<int64_t>  count   // Number of (class,theta,U)-voxels
             ){
    return cpu_seq::cylinder_projection(edt, C, voxel_size, d_min, d_max, theta_min, theta_max, bbox, Muvw, image, count);
}

void fill_implant_mask(const input_ndarray<mask_type> mask,
               int64_t offset,
               float voxel_size,
               const array<float,6> &bbox,
               float r_fraction,
               const matrix4x4 &Muvw,
               const input_ndarray<real_t> thetas,
               const input_ndarray<float> rsqr_maxs,
               output_ndarray<mask_type> solid_implant_mask,
               output_ndarray<float> profile) {
    return cpu_seq::fill_implant_mask(mask, offset, voxel_size, bbox, r_fraction, Muvw, thetas, rsqr_maxs, solid_implant_mask, profile);
}

void fill_implant_mask_pre(const input_ndarray<mask_type> mask,
               int64_t offset,
               float voxel_size,
               const array<float,6> &bbox,
               const matrix4x4 &Muvw,
               output_ndarray<real_t> thetas,
               output_ndarray<float> rsqr_maxs) {
    return cpu_seq::fill_implant_mask_pre(mask, offset, voxel_size, bbox, Muvw, thetas, rsqr_maxs);
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