/**
 * @file bitpacking.cc
 * Sequential CPU implementation of the bitpacking functions.
 */
#include "bitpacking.hh"

namespace cpu_seq {

    template <typename T>
    void encode(const uint8_t *mask, const uint64_t n, T *packed, const int verbose) {
        constexpr uint64_t T_bits = sizeof(T)*8;

        if (verbose >= 2) {
            std::cout << "Packing " << n << " bits into " << n/T_bits << " " << sizeof(T)*8 << "-bit integers" << std::endl;
        }

        #pragma omp parallel for
        for (uint64_t i = 0; i < n; i += T_bits) {
            T packed_value = 0;
            for (uint64_t j = 0; j < T_bits; j++) {
                packed_value |= (mask[i+j] & 1) << ((T_bits-1)-j);
            }
            packed[i/T_bits] = packed_value;
        }
    }

    template <typename T>
    void decode(const T *packed, const uint64_t n, mask_type *mask, const int verbose) {
        constexpr uint64_t T_bits = sizeof(T)*8;

        if (verbose >= 2) {
            std::cout << "Unpacking " << n/T_bits << " " << sizeof(T)*8 << "-bit integers into " << n << " bits" << std::endl;
        }

        #pragma omp parallel for
        for (uint64_t i = 0; i < n; i += T_bits) {
            T packed_value = packed[i/T_bits];
            for (uint64_t j = 0; j < T_bits; j++) {
                mask[i+j] = (packed_value >> ((T_bits-1)-j)) & 1;
            }
        }
    }

    template <typename T>
    void slice(const T *packed, const shape_t &total_shape, const shape_t &slice_shape,
               const shape_t &offset, T *slice, const int verbose) {
        auto [Nz, Ny, Nx] = total_shape;
        auto [sz, sy, sx] = slice_shape;
        auto [oz, oy, ox] = offset;

        if (verbose >= 2) {
            std::cout << "Extracting slice of shape (" << sz << ", " << sy << ", " << sx << ") from offset (" << oz << ", " << oy << ", " << ox << ")" << std::endl;
            std::cout << "Total shape is (" << Nz << ", " << Ny << ", " << Nx << ")" << std::endl;
        }

        #pragma omp parallel for
        for (uint64_t z = 0; z < sz; z++) {
            for (uint64_t y = 0; y < sy; y++) {
                for (uint64_t x = 0; x < sx; x++) {
                    // TODO Handle unalignment
                    uint64_t packed_offset = (oz+z)*Ny*Nx + (oy+y)*Nx + ox+x;
                    uint64_t slice_offset = z*sy*sx + y*sx + x;
                    slice[slice_offset] = packed[packed_offset];
                }
            }
        }
    }

}