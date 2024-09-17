/**
 * @file geometry.cc
 * @author Carl-Johannes Johnsen (carl-johannes@di.ku.dk)
 * @brief Geometry-related functions for the GPU implementation.
 * @version 0.1
 * @date 2024-09-17
 *
 * @copyright Copyright (c) 2024
 */
#include "geometry.hh"
#include "../cpu_seq/geometry.cc"

namespace gpu {

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
            output_ndarray<int64_t> count) {
        return cpu_seq::cylinder_projection(edt, C, voxel_size, d_min, d_max, theta_min, theta_max, bbox, Muvw, image, count);
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
        UNPACK_NUMPY(mask)

        const real_t *thetas_d = thetas.data;
        real_t theta_center = (thetas_d[1] + thetas_d[0]) / 2;
        ssize_t n_segments = rsqr_maxs.shape[0];
        const auto [U_min, U_max, V_min, V_max, W_min, W_max] = bbox;
        const float *rsqr_maxs_d = rsqr_maxs.data;
        float *profile_d = profile.data;

        #pragma acc data copyin(U_min, U_max, W_min, Muvw, mask_Nz, mask_Ny, mask_Nx, voxel_size, n_segments, bbox, theta_center, thetas_d[:2], rsqr_maxs_d[:n_segments]) copy(profile_d[:n_segments])
        {
            for (int64_t mask_buffer_start = 0; mask_buffer_start < mask_length; mask_buffer_start += acc_block_size<mask_type>) {
                const mask_type *mask_buffer = mask.data + mask_buffer_start;
                ssize_t mask_buffer_length = std::min(acc_block_size<mask_type>, mask_length-(ssize_t)mask_buffer_start);
                mask_type *solid_mask_buffer = solid_implant_mask.data + offset + mask_buffer_start;
                #pragma acc data copy(mask_buffer[:mask_buffer_length]) create(solid_mask_buffer[:mask_buffer_length]) copyout(solid_mask_buffer[:mask_buffer_length])
                {
                    #pragma acc parallel loop // reduction(+:profile_d[:n_segments])
                    for (int64_t flat_index = 0; flat_index < mask_buffer_length; flat_index++) {
                        int64_t
                            global_index = offset + mask_buffer_start + flat_index,
                            z = global_index / (mask_Ny * mask_Nx),
                            y = (global_index / mask_Nx) % mask_Ny,
                            x = global_index % mask_Nx;
                        mask_type mask_value = mask_buffer[flat_index];
                        std::array<real_t, 4> Xs = {
                            real_t(z) * voxel_size,
                            real_t(y) * voxel_size,
                            real_t(x) * voxel_size,
                            1 };

                        // Second pass does the actual work
                        auto [U,V,W,c] = hom_transform(Xs, Muvw);
                        float r_sqr = V*V + W*W;
                        float theta = std::atan2(V, W);
                        int U_i = int(std::floor((U - U_min) * real_t(n_segments-1) / (U_max - U_min)));

                        bool solid_mask_value = false;
                        if (U_i >= 0 && U_i < n_segments && W >= W_min) { // TODO: Full bounding box check?
                            solid_mask_value = mask_value | (r_sqr <= r_fraction * rsqr_maxs_d[U_i]);

                            if (theta >= thetas_d[0] && theta <= theta_center && r_sqr <= rsqr_maxs_d[U_i]) {
                                #pragma acc atomic update
                                profile_d[U_i] += solid_mask_value;
                            }
                        }

                        solid_mask_buffer[flat_index] = solid_mask_value;
                    }
                }
            }
        }
    }

    void fill_implant_mask_pre(const input_ndarray<mask_type> mask,
            int64_t offset,
            float voxel_size,
            const std::array<float, 6> &bbox,
            const matrix4x4 &Muvw,
            output_ndarray<real_t> thetas,
            output_ndarray<float> rsqr_maxs) {
        UNPACK_NUMPY(mask)

        real_t *thetas_d = thetas.data;

        if (offset == 0) {
            thetas_d[0] = real_t( M_PI);
            thetas_d[1] = real_t(-M_PI);
        }
        ssize_t n_segments = rsqr_maxs.shape[0];
        const auto [U_min, U_max, V_min, V_max, W_min, W_max] = bbox;
        float *rsqr_maxs_d = rsqr_maxs.data;

        #pragma acc data copyin(U_min, U_max, W_min, Muvw, mask_Nz, mask_Ny, mask_Nx, voxel_size, n_segments, bbox) copy(rsqr_maxs_d[:n_segments])
        {
            for (int64_t mask_buffer_start = 0; mask_buffer_start < mask_length; mask_buffer_start += acc_block_size<mask_type>) {
                ssize_t mask_buffer_length = std::min(acc_block_size<mask_type>, mask_length-(ssize_t)mask_buffer_start);
                ssize_t num_threads = std::min(mask_buffer_length, gpu_threads);
                const mask_type *mask_buffer = mask.data + mask_buffer_start;
                real_t
                    thetas_min = thetas_d[0],
                    thetas_max = thetas_d[1];
                #pragma acc data copyin(mask_buffer_start, mask_buffer[:mask_buffer_length]) copy(thetas_min, thetas_max)
                {
                    #pragma acc parallel loop reduction(max:thetas_max) reduction(min:thetas_min) reduction(max:rsqr_maxs_d[:n_segments])
                    for (int64_t thread = 0; thread < num_threads; thread++) {
                        for (int64_t thread_idx = 0; thread_idx < mask_buffer_length; thread_idx += gpu_threads) {
                            int64_t
                                flat_index = thread_idx + thread,
                                global_index = offset + mask_buffer_start + flat_index,
                                z = global_index / (mask_Ny * mask_Nx),
                                y = (global_index / mask_Nx) % mask_Ny,
                                x = global_index % mask_Nx;
                            mask_type mask_value = mask_buffer[flat_index];
                            std::array<real_t, 4> Xs = {
                                real_t(z) * voxel_size,
                                real_t(y) * voxel_size,
                                real_t(x) * voxel_size,
                                1
                            };

                            if (mask_value) {
                                auto [U,V,W,c] = hom_transform(Xs, Muvw);

                                real_t r_sqr = V*V + W*W;
                                real_t theta = std::atan2(V,W);

                                int U_i = int(std::floor((U - U_min) * real_t(n_segments-1) / (U_max - U_min)));

                                if (in_bbox({{U,V,W}}, bbox)) {
                                    rsqr_maxs_d[U_i] = std::max(rsqr_maxs_d[U_i], float(r_sqr));
                                    thetas_min = std::min(thetas_min, theta);
                                    thetas_max = std::max(thetas_max, theta);
                                } else {
                                    // Otherwise we've calculated it wrong!
                                }
                            }
                        }
                    }
                }
                thetas_d[0] = thetas_min;
                thetas_d[1] = thetas_max;
            }
        }
    }

    std::array<real_t, 9> inertia_matrix(const input_ndarray<mask_type> &mask, const std::array<real_t, 3> &cm) {
        return cpu_seq::inertia_matrix(mask, cm);
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
                    output_ndarray<real_t> plane_samples) {
        return cpu_seq::sample_plane(voxels, voxel_size, cm, u_axis, v_axis, bbox, plane_samples);
    }

    void zero_outside_bbox(const std::array<real_t, 9> &principal_axes,
                        const std::array<real_t, 6> &parameter_ranges,
                        const std::array<real_t, 3> &cm,
                        output_ndarray<mask_type> voxels) {
        return cpu_seq::zero_outside_bbox(principal_axes, parameter_ranges, cm, voxels);
    }

}
