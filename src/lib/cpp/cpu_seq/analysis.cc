#include "analysis.hh"

namespace cpu_seq {

    // Assumes that the mask and field are the same size
    float bic(const input_ndarray<bool> &mask, const input_ndarray<uint16_t> &field, const uint16_t threshold) {
        UNPACK_NUMPY(mask);
        UNPACK_NUMPY(field);

        uint64_t total = 0, count = 0;
        int64_t BLOCK_SIZE = 32;

        for (int64_t z_start = 0; z_start < mask_Nz; z_start += BLOCK_SIZE) {
            int64_t
                z_end = std::min(z_start + BLOCK_SIZE, mask_Nz),
                this_z = z_end - z_start,
                flat_start = z_start * mask_Ny * mask_Nx,
                flat_end = z_end * mask_Ny * mask_Nx,
                flat_size = flat_end - flat_start;

            const bool *mask_ptr = mask.data + flat_start;
            const uint16_t *field_ptr = field.data + flat_start;

            #pragma acc data copyin(mask_ptr[:flat_size], field_ptr[:flat_size]) copy(total, count)
            {
                PRAGMA(PARALLEL_TERM collapse(3) reduction(+:total,count)) \
                for (int64_t z = 0; z < this_z; z++) {
                    for (int64_t y = 0; y < mask_Ny; y++) {
                        for (int64_t x = 0; x < mask_Nx; x++) {
                            uint64_t flat_index = z * mask_Ny * mask_Nx + y * mask_Nx + x;
                            bool mask_val = mask_ptr[flat_index];
                            uint8_t is_close = field_ptr[flat_index] < threshold;
                            total += is_close;
                            count += mask_val & is_close;
                        }
                    }
                }
            }
        }

        return 1.0f - ((float)count / (float)total);
    }

}