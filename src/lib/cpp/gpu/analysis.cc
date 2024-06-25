#include "../cpu_seq/analysis.cc"

namespace gpu {

    void bic(const input_ndarray<bool> &mask, const input_ndarray<uint16_t> &field, const uint16_t threshold, output_ndarray<float> &output) {
        return cpu_seq::bic(mask, field, threshold, output);
    }

}