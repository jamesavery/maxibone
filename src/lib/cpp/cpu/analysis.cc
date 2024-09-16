/**
 * @file analysis.cc
 * Parallel CPU implementation of the BIC functions.
 */
#include "../cpu_seq/analysis.cc"

namespace cpu_par {

    void bic(const input_ndarray<bool> &voxels, const input_ndarray<uint16_t> &field, const input_ndarray<bool> &mask, uint16_t threshold, output_ndarray<float> &output) {
        // The implementation of this function is provided in the file `src/lib/cpp/cpu_seq/analysis.cc`.
        return cpu_seq::bic(voxels, field, mask, threshold, output);
    }

}