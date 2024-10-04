/**
 * @file analysis.cc
 * Sequential CPU implementation of the BIC functions.
 */
#include "analysis.hh"

namespace cpu_seq {

    void bic(const input_ndarray<bool> &voxels, const input_ndarray<uint16_t> &field, const input_ndarray<bool> &mask, uint16_t threshold, output_ndarray<float> &output, const int verbose) {
        // Unpack the input arrays.
        UNPACK_NUMPY(voxels);
        UNPACK_NUMPY(field);
        UNPACK_NUMPY(mask);
        UNPACK_NUMPY(output);

        int64_t
            BLOCK_SIZE  = 32,
            field_scale = voxels_Nz / field_Nz,
            mask_scale  = voxels_Nz / mask_Nz;

        float *output_ptr = output.data;

        #pragma acc data copyout(output_ptr[:voxels_Nz])
        for (int64_t z_start = 0; z_start < voxels_Nz; z_start += BLOCK_SIZE) {
            if (verbose >= 1) {
                printf("\rComputing BIC: %d%%", (int)(100.0 * (double)z_start / (double)voxels_Nz));
                fflush(stdout);
            }

            int64_t
                z_end = std::min(z_start + BLOCK_SIZE, voxels_Nz),
                this_z = z_end - z_start,

                voxels_flat_start = z_start * voxels_Ny * voxels_Nx,
                voxels_flat_end   = z_end   * voxels_Ny * voxels_Nx,
                __attribute__((unused)) voxels_flat_size = voxels_flat_end - voxels_flat_start,

                field_flat_start = (std::min(z_start / field_scale, field_Nz-1)) * field_Ny * field_Nx,
                field_flat_end   = (std::min(z_end   / field_scale, field_Nz  )) * field_Ny * field_Nx,
                __attribute__((unused)) field_flat_size = field_flat_end - field_flat_start,

                mask_flat_start = (std::min(z_start / mask_scale, mask_Nz-1)) * mask_Ny * mask_Nx,
                mask_flat_end   = (std::min(z_end   / mask_scale, mask_Nz  )) * mask_Ny * mask_Nx,
                __attribute__((unused)) mask_flat_size = mask_flat_end - mask_flat_start;

            const bool *voxels_ptr    = voxels.data + voxels_flat_start;
            const uint16_t *field_ptr =  field.data +  field_flat_start;
            const bool *mask_ptr      =   mask.data +   mask_flat_start;

            #pragma acc data copyin(voxels_ptr[:voxels_flat_size], field_ptr[:field_flat_size], mask_ptr[:mask_flat_size])
            {
                PRAGMA(PARALLEL_TERM)
                for (int64_t z = 0; z < this_z; z++) {
                    uint64_t total = 0, count = 0;
                    for (int64_t y = 0; y < voxels_Ny; y++) {
                        for (int64_t x = 0; x < voxels_Nx; x++) {
                            int64_t
                                mz = z / mask_scale,
                                my = y / mask_scale,
                                mx = x / mask_scale;
                            uint64_t mask_flat_index = mz * mask_Ny * mask_Nx + my * mask_Nx + mx;
                            bool mask_val = mask_ptr[mask_flat_index];

                            if (mask_val) {
                                uint64_t voxels_flat_index = z * voxels_Ny * voxels_Nx + y * voxels_Nx + x;
                                bool voxels_val = voxels_ptr[voxels_flat_index];

                                int64_t
                                    fz = z / field_scale,
                                    fy = y / field_scale,
                                    fx = x / field_scale;
                                uint64_t field_flat_index = fz * field_Ny * field_Nx + fy * field_Nx + fx;
                                uint16_t field_val = field_ptr[field_flat_index];

                                uint8_t is_close = (field_val < threshold) & (field_val > 0);
                                total += is_close;
                                count += voxels_val & is_close;
                            }
                        }
                    }
                    output_ptr[z_start + z] = 1.0f - ((float)count / (float)total);
                }
            }
        }

        if (verbose >= 1) {
            printf("\rComputing BIC: 100%%\n");
            fflush(stdout);
        }
    }

}