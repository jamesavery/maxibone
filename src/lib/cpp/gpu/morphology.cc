#include "morphology.hh"
#include "datatypes.hh"

namespace gpu {

template <typename Op, bool neutral>
void morphology_3d_sphere(
        const mask_type *voxels,
        const int64_t radius,
        const int64_t N[3],
        const int64_t strides[3],
        mask_type *result) {
#ifdef _OPENACC
    Op op;
    const int32_t sqradius = (int32_t)radius * (int32_t)radius;
    const int32_t nz = (int32_t)N[0], ny = (int32_t)N[1], nx = (int32_t)N[2];
    const int64_t sz = strides[0], sy = strides[1], sx = strides[2];

    #pragma acc data copyin(voxels[:nz*ny*nx]) copyout(result[:nz*ny*nx])
    {
        #pragma acc parallel loop collapse(3)
        for (int32_t z = 0; z < nz; z++) {
            for (int32_t y = 0; y < ny; y++) {
                for (int32_t x = 0; x < nx; x++) {
                    // Compute boundaries
                    int64_t flat_index = (int64_t)z*sz + (int64_t)y*sy + (int64_t)x*sx;

                    // Apply the spherical kernel
                    bool value = neutral;

                    for (int32_t pz = -radius; pz <= radius; pz++) {
                        for (int32_t py = -radius; py <= radius; py++) {
                            for (int32_t px = -radius; px <= radius; px++) {
                                bool within = px*px + py*py + pz*pz <= sqradius; // sphere kernel
                                within &= (z+pz >= 0);
                                within &= (z+pz < nz);
                                within &= (y+py >= 0);
                                within &= (y+py < ny);
                                within &= (x+px >= 0);
                                within &= (x+px < nx);
                                int64_t offset = (int64_t)pz*sz + (int64_t)py*sy + (int64_t)px*sx;

                                value = within ? op(value, voxels[flat_index+offset]) : value;
                            }
                        }
                    }

                    // Store the results
                    result[flat_index] = value;
                }
            }
        }
    }
#else
    throw runtime_error("Library wasn't compiled with OpenACC.");
#endif
}

// Hardcoded radius 16, as this is the general case, and hardcoding allows for more optimizations.
template <typename Op, bool neutral>
void morphology_3d_sphere_r16(
        const mask_type *voxels,
        const int64_t N[3],
        const int64_t strides[3],
        mask_type *result) {
#ifdef _OPENACC
    Op op;
    constexpr int32_t radius = 16;
    const int32_t sqradius = (int32_t)radius * (int32_t)radius;
    const int32_t nz = (int32_t)N[0], ny = (int32_t)N[1], nx = (int32_t)N[2];
    const int64_t sz = strides[0], sy = strides[1], sx = strides[2];

    #pragma acc data copyin(voxels[:nz*ny*nx]) copyout(result[:nz*ny*nx])
    {
        #pragma acc parallel loop collapse(3)
        for (int32_t z = 0; z < nz; z++) {
            for (int32_t y = 0; y < ny; y++) {
                for (int32_t x = 0; x < nx; x++) {
                    // Compute boundaries
                    int64_t flat_index = (int64_t)z*sz + (int64_t)y*sy + (int64_t)x*sx;

                    // Apply the spherical kernel
                    bool value = neutral;

                    #pragma omp simd collapse(3) reduction(op:value)
                    for (int32_t pz = -radius; pz <= radius; pz++) {
                        for (int32_t py = -radius; py <= radius; py++) {
                            for (int32_t px = -radius; px <= radius; px++) {
                                bool within = px*px + py*py + pz*pz <= sqradius; // sphere kernel
                                within &= (z+pz >= 0);
                                within &= (z+pz < nz);
                                within &= (y+py >= 0);
                                within &= (y+py < ny);
                                within &= (x+px >= 0);
                                within &= (x+px < nx);
                                int64_t offset = (int64_t)pz*sz + (int64_t)py*sy + (int64_t)px*sx;

                                value = within ? op(value, voxels[flat_index+offset]) : value;
                            }
                        }
                    }

                    // Store the results
                    result[flat_index] = value;
                }
            }
        }
    }
#else
    throw runtime_error("Library wasn't compiled with OpenACC.");
#endif
}

} // namespace gpu