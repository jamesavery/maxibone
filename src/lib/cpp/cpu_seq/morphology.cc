/**
 * @file morphology.hh
 * @author Carl-Johannes Johnsen (carl-johannes@di.ku.dk)
 * @brief CPU sequential implementation of the morphology operations on 3D masks.
 * @version 0.1
 * @date 2024-09-17
 *
 * @copyright Copyright (c) 2024
 */
#include "morphology.hh"

namespace cpu_seq {

template <typename Op, bool neutral>
void morphology_3d_sphere(
        const mask_type *voxels,
        const int64_t radius,
        const int64_t N[3],
        const int64_t strides[3],
        mask_type *result) {
    Op op;
    int64_t sqradius = radius * radius;

    #pragma omp parallel for collapse(3)
    for (int64_t z = 0; z < N[0]; z++) {
        for (int64_t y = 0; y < N[1]; y++) {
            for (int64_t x = 0; x < N[2]; x++) {
                // Compute boundaries
                int64_t flat_index = z*strides[0] + y*strides[1] + x*strides[2];
                int64_t X[3] = {z, y, x};
                int64_t limits[6];
                for (int axis = 0; axis < 3; axis++) {
                    limits[(axis*2)] = -min(radius, X[axis]);
                    limits[(axis*2)+1] = min(radius, N[axis] - X[axis] - 1);
                }

                // Apply the spherical kernel
                bool value = neutral;
                for (int64_t pz = limits[0]; pz <= limits[1]; pz++) {
                    for (int64_t py = limits[2]; py <= limits[3]; py++) {
                        for (int64_t px = limits[4]; px <= limits[5]; px++) {
                            // TODO exact match with ndimage
                            bool within = px*px + py*py + pz*pz <= sqradius; // sphere kernel
                            int64_t offset = pz*strides[0] + py*strides[1] + px*strides[2];
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

template <typename Op, bool neutral>
void morphology_3d_sphere_r16(
        const mask_type *voxels,
        const int64_t N[3],
        const int64_t strides[3],
        mask_type *result) {
    Op op;
    constexpr int32_t radius = 16;
    const int32_t sqradius = (int32_t)radius * (int32_t)radius;
    const int32_t nz = (int32_t)N[0], ny = (int32_t)N[1], nx = (int32_t)N[2];
    const int64_t sz = strides[0], sy = strides[1], sx = strides[2];

    #pragma omp parallel for collapse(3)
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

template <uint32_t op(uint32_t,uint32_t), uint32_t reduce(uint32_t,uint32_t), uint32_t neutral>
void morphology_3d_sphere_bitpacked(
        const uint32_t *voxels,
        const int64_t radius,
        const int64_t N[3],
        const int64_t strides[3],
        uint32_t *result) {
    // TODO assumes that Nx is a multiple of 32, which is true for scale <= 4
    int64_t
        k = radius*2 + 1,
        sqradius = radius * radius;

    // TODO handle k > 32
    // TODO templated construction? Has to 'hardcode' a radius, but that is beneficial anyways.
    // Create the kernel
    uint32_t *kernel = (uint32_t*) malloc(k*k*sizeof(uint32_t));

    #pragma omp parallel for collapse(2)
    for (int64_t z = -radius; z <= radius; z++) {
        for (int64_t y = -radius; y <= radius; y++) {
            uint32_t row = 0;
            for (int64_t x = 0; x < 32; x++) {
                uint32_t element = (x-radius)*(x-radius) + y*y + z*z <= sqradius;
                row |= element << (31 - x);
            }
            kernel[(z+radius)*k + y+radius] = row;
        }
    }

    #pragma omp parallel for collapse(3)
    for (int64_t z = 0; z < N[0]; z++) {
        for (int64_t y = 0; y < N[1]; y++) {
            for (int64_t x = 0; x < N[2]/32; x++) {
                // Compute boundaries
                int64_t flat_index = z*strides[0] + y*strides[1] + x*strides[2];
                int64_t X[3] = {z, y, x};
                int64_t limits[6];
                for (int axis = 0; axis < 3; axis++) {
                    limits[(axis*2)] = -min(radius, X[axis]);
                    limits[(axis*2)+1] = min(radius, N[axis] - X[axis] - 1);
                }

                // Apply the spherical kernel
                uint32_t value = neutral;
                for (int64_t pz = limits[0]; pz <= limits[1]; pz++) {
                    for (int64_t py = limits[2]; py <= limits[3]; py++) {
                        int64_t this_flat_index = flat_index + pz*strides[0] + py*strides[1];
                        uint32_t
                            left = x == 0 ? neutral : voxels[this_flat_index - 1],
                            middle = voxels[this_flat_index],
                            right = x == (N[2]/32)-1 ? neutral : voxels[this_flat_index + 1],
                            kernel_row = kernel[(pz+radius)*k + (py+radius)];

                        uint32_t this_row = 0;
                        for (int64_t px = 0; px < 32; px++) {
                            uint32_t this_x = 0 |
                                (left   << (32 - radius + px)) |
                                (middle >> (radius - px))      |
                                (middle << (-radius + px))     |
                                (right  >> (32 + radius - px));
                            this_x = reduce(this_x, kernel_row);
                            this_row |= this_x << (31 - px);
                        }
                        value = op(value, this_row);
                    }
                }

                // Store the results
                result[flat_index] = value;
            }
        }
    }
}

} // namespace cpu_seq