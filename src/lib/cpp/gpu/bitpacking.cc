#include "bitpacking.hh"
#include "datatypes.hh"

namespace gpu {

    // TODO Introduce bounding; currently assumes that n is a multiple of T_bits*vec_size*worker_size

    // The following GPU specific optimizations have been applied:
    // 1. Coalesced gloabl memory access (A warp reads consecutive global memory)
    // 2. Caching in shared memory (To allow for faster small transactions) (First edition had padding to resolve bank conflicts)
    // 3. Coalesced shared memory access (A warp reads consecutive shared memory)
    // 4. Resolving bank conflicts in shared memory (By cyclically accessing a byte with an offset, rather than consecutive bytes)

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
                        // Load mask into shared memory with coalesced access to mask
                        #pragma acc loop vector
                        for (uint64_t k = 0; k < vec_size; k++) {
                            for (uint64_t l = 0; l < T_bits; l++) {
                                local[l*vec_size + k] = mask[offset + l*vec_size + k];
                            }
                        }

                        // Pack bits into T
                        #pragma acc loop vector
                        for (uint64_t k = 0; k < vec_size; k++) {
                            T packed_value = 0;
                            for (uint64_t l = 0; l < T_bits; l++) {
                                uint32_t val;
                                uint64_t bank_offset = ((k+l) % T_bits);
                                val = local[k*T_bits + bank_offset] & 1;
                                packed_value |= val << ((T_bits-1)-(bank_offset));
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
        constexpr uint64_t
            T_bits = sizeof(T)*8,
            vec_size = 32,
            worker_size = 1;

        #pragma acc data copyin(packed[0:n/T_bits]) copyout(mask[0:n])
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
                        #pragma acc loop vector
                        for (uint64_t k = 0; k < vec_size; k++) {
                            T packed_value = packed[offset/T_bits + k];
                            for (uint64_t l = 0; l < T_bits; l++) {
                                // Option 0
                                local[k*T_bits + l] = (packed_value >> ((T_bits-1)-l)) & 1;
                                // Option 1
                                //local[l*vec_size + k] = (packed_value >> ((T_bits-1)-l)) & 1;
                            }
                        }

                        #pragma acc loop vector
                        for (uint64_t k = 0; k < vec_size; k++) {
                            for (uint64_t l = 0; l < T_bits; l++) {
                                // Option 0
                                mask[offset + l*vec_size + k] = local[l*vec_size + k];
                                // Option 1
                                //mask[offset + l*vec_size + k] = local[k*T_bits + l];
                            }
                        }
                    }
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