/**
 * @file geometry.cc
 * @author Carl-Johannes Johnsen (carl-johannes@di.ku.dk)
 * @brief Geometry-related functions for the CPU parallel implementation.
 * @version 0.1
 * @date 2024-09-17
 *
 * @copyright Copyright (c) 2024
 */
#include "geometry.hh"
#include "../cpu_seq/geometry.cc"

namespace cpu_par {

    std::array<real_t, 3> center_of_mass(const input_ndarray<mask_type> &mask) {
        return cpu_seq::center_of_mass(mask);
    }

    void center_of_masses(const input_ndarray<uint64_t> &mask, output_ndarray<real_t> &output) {
        return cpu_seq::center_of_masses(mask, output);
    }

    void compute_front_mask(const input_ndarray<mask_type> solid_implant,
            const float voxel_size,
            const matrix4x4 &Muvw,
            std::array<float, 6> bbox,
            output_ndarray<mask_type> front_mask) {
        return cpu_seq::compute_front_mask(solid_implant, voxel_size, Muvw, bbox, front_mask);
    }

    void compute_front_back_masks(const mask_type *mask, const shape_t &shape, const float voxel_size, const float *E, const float *cm, const float *cp, const float *UVWp, mask_type *front_mask, mask_type *back_mask, mask_type *implant_shell_mask, mask_type *solid_implant) {
        return cpu_seq::compute_front_back_masks(mask, shape, voxel_size, E, cm, cp, UVWp, front_mask, back_mask, implant_shell_mask, solid_implant);
    }

    void cylinder_projection(const input_ndarray<float> edt,
            const input_ndarray<uint8_t> C,
            float voxel_size,
            float d_min, float d_max,
            float theta_min, float theta_max,
            std::array<float, 6> bbox,
            const matrix4x4 &Muvw,
            output_ndarray<float> image,
            output_ndarray<int64_t> count,
            const int verbose) {
        return cpu_seq::cylinder_projection(edt, C, voxel_size, d_min, d_max, theta_min, theta_max, bbox, Muvw, image, count, verbose);
    }

    void fill_implant_mask(const input_ndarray<mask_type> mask,
            int64_t offset,
            float voxel_size,
            const std::array<float, 6> &bbox,
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
            const std::array<float, 6> &bbox,
            const matrix4x4 &Muvw,
            output_ndarray<real_t> thetas,
            output_ndarray<float> rsqr_maxs) {
        return cpu_seq::fill_implant_mask_pre(mask, offset, voxel_size, bbox, Muvw, thetas, rsqr_maxs);
    }

    std::array<real_t, 9> inertia_matrix(const input_ndarray<mask_type> &mask, const std::array<real_t, 3> &cm, const int verbose) {
        return cpu_seq::inertia_matrix(mask, cm, verbose);
    }

    void inertia_matrices(const input_ndarray<uint64_t> &mask, const input_ndarray<real_t> &cms, output_ndarray<real_t> &output) {
        return cpu_seq::inertia_matrices(mask, cms, output);
    }

    void integrate_axes(const input_ndarray<mask_type> &mask,
            const std::array<real_t, 3> &x0,
            const std::array<real_t, 3> &v_axis,
            const std::array<real_t, 3> &w_axis,
            const real_t v_min, const real_t w_min,
            output_ndarray<uint64_t> output) {
        return cpu_seq::integrate_axes(mask, x0, v_axis, w_axis, v_min, w_min, output);
    }

    void outside_ellipsoid(const input_ndarray<uint64_t> &voxels, const input_ndarray<real_t> &cms, const input_ndarray<real_t> &abc, output_ndarray<uint64_t> &errors) {
        return cpu_seq::outside_ellipsoid(voxels, cms, abc, errors);
    }

    template <typename T>
    void sample_plane(const input_ndarray<T> &voxels,
            const real_t voxel_size,
            const std::array<real_t, 3> cm,
            const std::array<real_t, 3> u_axis,
            const std::array<real_t, 3> v_axis,
            const std::array<real_t, 4> bbox,
            output_ndarray<real_t> plane_samples,
            const int verbose) {
        return cpu_seq::sample_plane(voxels, voxel_size, cm, u_axis, v_axis, bbox, plane_samples, verbose);
    }

    void zero_outside_bbox(const std::array<real_t, 9> &principal_axes,
            const std::array<real_t, 6> &parameter_ranges,
            const std::array<real_t, 3> &cm,
            output_ndarray<mask_type> voxels) {
        return cpu_seq::zero_outside_bbox(principal_axes, parameter_ranges, cm, voxels);
    }

}