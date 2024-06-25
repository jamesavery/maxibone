#include "analysis.hh"

namespace cpu_seq {

    // Assumes that the mask and field are the same size
    void bic(const input_ndarray<bool> &mask, const input_ndarray<uint16_t> &field, const uint16_t threshold, output_ndarray<float> &output) {
        UNPACK_NUMPY(mask);
        UNPACK_NUMPY(field);

        int64_t BLOCK_SIZE = 32;

        #pragma acc data copy(output.data[:mask_Nz])
        for (int64_t z_start = 0; z_start < mask_Nz; z_start += BLOCK_SIZE) {
            int64_t
                z_end = std::min(z_start + BLOCK_SIZE, mask_Nz),
                this_z = z_end - z_start,
                flat_start = z_start * mask_Ny * mask_Nx,
                flat_end = z_end * mask_Ny * mask_Nx,
                __attribute__((unused)) flat_size = flat_end - flat_start;

            const bool *mask_ptr = mask.data + flat_start;
            const uint16_t *field_ptr = field.data + flat_start;

            #pragma acc data copyin(mask_ptr[:flat_size], field_ptr[:flat_size])
            {
                PRAGMA(PARALLEL_TERM)
                for (int64_t z = 0; z < this_z; z++) {
                    uint64_t total = 0, count = 0;
                    for (int64_t y = 0; y < mask_Ny; y++) {
                        for (int64_t x = 0; x < mask_Nx; x++) {
                            uint64_t flat_index = z * mask_Ny * mask_Nx + y * mask_Nx + x;
                            bool mask_val = mask_ptr[flat_index];
                            uint16_t field_val = field_ptr[flat_index];
                            uint8_t is_close = (field_val < threshold) & (field_val > 0);
                            total += is_close;
                            count += mask_val & is_close;
                        }
                    }
                    output.data[z_start + z] = 1.0f - ((float)count / (float)total);
                }
            }
        }
    }

}