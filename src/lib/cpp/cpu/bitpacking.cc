#include "bitpacking.hh"

#include "../cpu_seq/bitpacking.cc"

namespace cpu_par {

    template <typename T>
    void encode(const uint8_t *mask, const uint64_t n, T *packed) {
        cpu_seq::encode<T>(mask, n, packed);
    }

    template <typename T>
    void decode(const T *packed, const uint64_t n, mask_type *mask) {
        cpu_seq::decode<T>(packed, n, mask);
    }

    template <typename T>
    void slice(const T *packed, const shape_t &total_shape, const shape_t &slice_shape,
               const shape_t &offset, T *slice) {
        cpu_seq::slice<T>(packed, total_shape, slice_shape, offset, slice);
    }

}