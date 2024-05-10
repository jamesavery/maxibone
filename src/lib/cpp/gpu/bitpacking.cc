#include "bitpacking.hh"
#include "datatypes.hh"

namespace gpu {

    // TODO Introduce bounding; currently assumes that n is a multiple of T_bits*vec_size*worker_size

    template <typename T>
    void encode(const uint8_t *mask, const uint64_t n, T *packed) {
        constexpr uint64_t
            T_bits = sizeof(T)*8,
            vec_size = 32,
            worker_size = 1;

        #pragma acc data copyin(mask[0:n]) copyout(packed[0:n/T_bits])
        #pragma acc parallel vector_length(vec_size) num_workers(worker_size)
        {
            #pragma acc loop gang
            for (uint64_t i = 0; i < n; i += worker_size*vec_size*T_bits) {
                #pragma acc loop worker
                for (uint64_t j = 0; j < worker_size; j++) {
                    uint32_t local[vec_size*T_bits]; // Shared memory
                    int64_t offset = i + j*vec_size*T_bits;
                    #pragma acc cache(local)
                    {
                        // TODO Option 0 produces load bank conflicts in the second pass, whereas option 1 produces store bank conflicts in the first pass.
                        // Load mask into shared memory with coalesced access to mask
                        #pragma acc loop vector
                        for (uint64_t k = 0; k < vec_size; k++) {
                            for (uint64_t l = 0; l < T_bits; l++) {
                                // 0 Coalesced:
                                //local[l*vec_size + k] = mask[offset + l*vec_size + k];
                                // 1 Coalesced read, uncoalesced write:
                                local[k*T_bits + l] = (uint32_t) mask[offset + l*vec_size + k];
                            }
                        }

                        // Pack bits into T
                        #pragma acc loop vector
                        for (uint64_t k = 0; k < vec_size; k++) {
                            T packed_value = 0;
                            for (uint64_t l = 0; l < T_bits; l++) {
                                // 0 Coalesced:
                                //packed_value |= (local[k*T_bits + l] & 1) << ((T_bits-1)-l);
                                // 1 Coalesced read, uncoalesced write:
                                packed_value |= (local[l*vec_size + k] & 1) << ((T_bits-1)-l);
                            }
                            packed[offset/T_bits + k] = packed_value;
                        }
                    }
                }
            }
        }
    }

    template <typename T>
    void decode(const T *packed, const uint64_t n, mask_type *mask) {
        constexpr uint64_t T_bits = sizeof(T)*8;

        #pragma acc data copyin(packed[0:n/T_bits]) copyout(mask[0:n])
        {
            #pragma acc parallel loop
            for (uint64_t i = 0; i < n; i += T_bits) {
                T packed_value = packed[i/T_bits];
                for (uint64_t j = 0; j < T_bits; j++) {
                    mask[i+j] = (packed_value >> ((T_bits-1)-j)) & 1;
                }
            }
        }
    }

    template <typename T>
    void slice(const T *packed, const shape_t &total_shape, const shape_t &slice_shape,
               const shape_t &offset, T *slice) {
        auto [Nz, Ny, Nx] = total_shape;
        auto [sz, sy, sx] = slice_shape;
        auto [oz, oy, ox] = offset;

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