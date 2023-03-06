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

void fill_implant_mask(const input_ndarray<mask_type> mask,
               float voxel_size,
               const array<float,6> &bbox,
               float r_fraction,
               const matrix4x4 &Muvw,
               output_ndarray<mask_type> solid_implant_mask,
               output_ndarray<float> rsqr_maxs,
               output_ndarray<float> profile) {
    UNPACK_NUMPY(mask)

    real_t theta_min = real_t(M_PI), theta_max = real_t(-M_PI);
    ssize_t n_segments = rsqr_maxs.shape[0];
    const auto [U_min, U_max, V_min, V_max, W_min, W_max] = bbox;
    float     *rsqr_maxs_d     = rsqr_maxs.data;
    float     *profile_d       = profile.data;

    #pragma acc data copyin(U_min) create(rsqr_maxs_d[:n_segments], profile_d[:n_segments]) copyout(rsqr_maxs_d[:n_segments], profile_d[:n_segments])
    {
        for (int64_t mask_buffer_start = 0; mask_buffer_start < mask_length; mask_buffer_start += acc_block_size<mask_type>) {
            ssize_t mask_buffer_length = min(acc_block_size<mask_type>, mask_length-mask_buffer_start);
            mask_type *mask_buffer = (mask_type *) mask.data + mask_buffer_start;
            #pragma acc data copy(mask_buffer[:mask_buffer_length])
            {
                #pragma acc parallel loop
                for (int64_t flat_index = 0; flat_index < mask_buffer_length; flat_index++) {
                    int64_t
                        global_index = mask_buffer_start + flat_index,
                        z = global_index / (mask_Ny * mask_Nx),
                        y = (global_index / mask_Nx) % mask_Ny,
                        x = global_index % mask_Nx;
                    mask_type mask_value = mask_buffer[flat_index];
                    std::array<real_t, 4> Xs = {
                        real_t(x) * voxel_size,
                        real_t(y) * voxel_size,
                        real_t(z) * voxel_size,
                        1 };

                    if (mask_value) {
                        auto [U,V,W,c] = hom_transform(Xs, Muvw);

                        real_t r_sqr = V*V+W*W;
                        real_t theta = atan2(V,W);

                        int U_i = int(floor((U-U_min)*real_t(n_segments-1)/(U_max-U_min)));

                        if ( in_bbox(U,V,W,bbox) ) {
                            rsqr_maxs_d[U_i] = max(rsqr_maxs_d[U_i], float(r_sqr));
                            theta_min = min(theta_min, theta);
                            theta_max = max(theta_max, theta);
                        } else {
                            // Otherwise we've calculated it wrong!
                        }
                    }
                }
            }
        }

        double theta_center = (theta_max+theta_min)/2;

        for (int64_t mask_buffer_start = 0; mask_buffer_start < mask_length; mask_buffer_start += acc_block_size<mask_type>) {
            mask_type *mask_buffer = (mask_type *) mask.data + mask_buffer_start;
            ssize_t mask_buffer_length = min(acc_block_size<mask_type>, mask_length-mask_buffer_start);
            mask_type *solid_mask_buffer = solid_implant_mask.data + mask_buffer_start;
            #pragma acc data copy(mask_buffer[:mask_buffer_length]) create(solid_mask_buffer[:mask_buffer_length]) copyout(solid_mask_buffer[:mask_buffer_length])
            {
                #pragma acc parallel loop
                for (int64_t flat_index = 0; flat_index < mask_buffer_length; flat_index++) {
                    int64_t
                        global_index = mask_buffer_start + flat_index,
                        z = global_index / (mask_Ny * mask_Nx),
                        y = (global_index / mask_Nx) % mask_Ny,
                        x = global_index % mask_Nx;
                    mask_type mask_value = mask_buffer[flat_index];
                    std::array<real_t, 4> Xs = {
                        real_t(x) * voxel_size,
                        real_t(y) * voxel_size,
                        real_t(z) * voxel_size,
                        1 };

                    // Second pass does the actual work
                    auto [U,V,W,c] = hom_transform(Xs,Muvw);
                    float r_sqr = V*V+W*W;
                    float theta = atan2(V,W);
                    int U_i = int(floor((U-U_min)*real_t(n_segments-1)/(U_max-U_min)));

                    bool solid_mask_value = false;
                    if (U_i >= 0 && U_i < n_segments && W >= W_min) { // TODO: Full bounding box check?
                        solid_mask_value = mask_value | (r_sqr <= r_fraction*rsqr_maxs_d[U_i]);

                        if (theta >= theta_min && theta <= theta_center && r_sqr <= rsqr_maxs_d[U_i]) {
                            ATOMIC()
                            profile_d[U_i] += solid_mask_value;
                        }
                    }

                    solid_mask_buffer[flat_index] = solid_mask_value;
                }
            }
        }
    }
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

void zero_outside_bbox(const array<real_t,9> &principal_axes,
                       const array<real_t,6> &parameter_ranges,
                       const array<real_t,3> &cm,
                       output_ndarray<mask_type> voxels) {
    return cpu_seq::zero_outside_bbox(principal_axes, parameter_ranges, cm, voxels);
}

}