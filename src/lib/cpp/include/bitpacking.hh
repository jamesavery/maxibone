/**
 * @file bitpacking.hh
 * Functions for bitpacking and bitunpacking.
 * Bitpacking is the process of storing a bool as 1 bit, rather than a byte.
 */
#ifndef bitpacking_h
#define bitpacking_h

#include "datatypes.hh"

namespace NS {

    /**
     * Pack the given `mask` of length `n` into `packed` of type `T`.
     * `T` should be an unsigned integer type: uint8_t, uint16_t, uint32_t, uint64_t.
     * It is assumed that `packed` is preallocated to the correct size, which is `n//(sizeof(T)*8)`.
     *
     * @param mask The mask to pack.
     * @param n The length of the mask.
     * @param packed The packed mask.
     * @tparam T The type of the packed mask.
     */
    template <typename T>
    void encode(const uint8_t *mask, const uint64_t n, T *packed);

    /**
     * Unpack the given `packed` of length `n` into `mask`.
     * It is the inverse operation of `encode`.
     * `T` should be an unsigned integer type: uint8_t, uint16_t, uint32_t, uint64_t.
     * It is assumed that `mask` is preallocated to the correct size, which is `n*8`.
     *
     * @param packed The packed mask.
     * @param n The length of the packed mask.
     * @param mask The unpacked mask.
     * @tparam T The type of the packed mask.
     */
    template <typename T>
    void decode(const T *packed, const uint64_t n, mask_type *mask);

    /**
     * Extract a slice (or chunk) of the bitpacked data `packed`, into `slice`.
     * The size of the slice is given by `slice_shape`, and the offset of the slice is given by `offset`.
     * The total shape of the data `packed` is given by `total_shape`.
     *
     * @param packed The bitpacked data.
     * @param total_shape The total shape of `packed`.
     * @param slice_shape The shape of the slice.
     * @param offset The offset of the slice.
     * @param slice The extracted slice.
     * @tparam T The type of the packed data.
     */
    template <typename T>
    void slice(const T *packed, const shape_t &total_shape, const shape_t &slice_shape, const shape_t &offset, T *slice);

} // namespace NS

#endif // bitpacking_h