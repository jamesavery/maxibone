#include "../cpu_seq/analysis.cc"

namespace cpu_par {

    float bic(const input_ndarray<bool> &mask, const input_ndarray<uint16_t> &field, const uint16_t threshold) {
        return cpu_seq::bic(mask, field, threshold);
    }

}