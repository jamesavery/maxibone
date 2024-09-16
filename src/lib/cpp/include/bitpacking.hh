/**
 * @file bitpacking.hh
 * Functions for bitpacking and bitunpacking.
 * Bitpacking is the process of storing a bool as 1 bit, rather than a byte.
 */
#ifndef bitpacking_h
#define bitpacking_h

#include "datatypes.hh"

namespace NS {

    template <typename T>
    void encode(const uint8_t *mask, const uint64_t n, T *packed);

    template <typename T>
    void decode(const T *packed, const uint64_t n, mask_type *mask);

    template <typename T>
    void slice(const T *packed, const shape_t &total_shape, const shape_t &slice_shape, const shape_t &offset, T *slice);

}

#endif // bitpacking_h