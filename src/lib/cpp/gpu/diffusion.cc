#include "diffusion.hh"

#include <iostream>

namespace gpu {

    void diffusion_core(const float *__restrict__ input, const float *__restrict__ kernel, float *__restrict__ output, const shape_t &N, const int64_t dim, const int64_t radius, const int64_t padding) {
        #pragma acc parallel loop collapse(3) present(input, kernel, output)
        for (int64_t i = 0; i < N.z+padding; i++) {
            for (int64_t j = 0; j < N.y+padding; j++) {
                for (int64_t k = 0; k < N.x+padding; k++) {
                    const int64_t
                        X[3] = {i, j, k},
                        stride[3] = {(N.y+padding)*(N.x+padding), N.x+padding, 1},
                        Ns[3] = {N.z+padding, N.y+padding, N.x+padding},
                        ranges[2] = {
                            -std::min(radius, X[dim]), std::min(radius, Ns[dim]-X[dim]-1)
                        },
                        output_index = i*stride[0] + j*stride[1] + k*stride[2];

                    float sum = 0.0f;

                    for (int64_t r = -radius; r <= radius; r++) {
                        const int64_t input_index = output_index + r*stride[dim];
                        // If the loop ranges guards the accesses
                        //float val = input[input_index];
                        //sum += val * kernel[radius+r];

                        // Branch free version
                        bool cond = r >= ranges[0] && r <= ranges[1];
                        int32_t mask = -cond;
                        mask &= mask >> 31;
                        int64_t index = (mask & input_index) | (~mask & output_index);
                        float val = input[index];

                        // Branch free version - pointer casting
                        int32_t *vali = (int32_t*) &val;
                        *vali &= mask;

                        // Branch free version - memcpy
                        //int32_t vali;
                        //std::memcpy(&vali, &val, sizeof(val));
                        //vali &= mask;
                        //std::memcpy(&val, &vali, sizeof(val));

                        // Branch free version - union
                        //raw32 val32;
                        //val32.f = val;
                        //val32.i &= mask;
                        //val = val32.f;

                        sum += val * kernel[radius+r];

                        // Original:
                        //float val = cond ? input[input_index] : 0.0f;
                        //sum += val * kernel[radius+r];
                    }

                    output[output_index] = sum;
                }
            }
        }
    }

    void convert_float_to_uint8(const float *__restrict__ src, uint8_t *__restrict__ dst, const int64_t total_flat_size) {
        #pragma acc parallel loop
        for (int64_t i = 0; i < total_flat_size; i++) {
            dst[i] = (uint8_t) std::floor(src[i] * 255.0f);
        }
    }

    void convert_uint8_to_float(const uint8_t *__restrict__ src, float *__restrict__ dst, const int64_t total_flat_size) {
        #pragma acc parallel loop
        for (int64_t i = 0; i < total_flat_size; i++) {
            dst[i] = src[i] > 0 ? 1.0f : 0.0f;
        }
    }

    void illuminate(const uint8_t *__restrict__ mask, float *__restrict__ output, const int64_t local_flat_size) {
        #pragma acc parallel loop
        for (int64_t i = 0; i < local_flat_size; i++) {
            if (mask[i] > 0) {
                output[i] = 1.0f;
            }
        }
    }

    void diffusion_in_memory(const uint8_t *__restrict__ voxels, const shape_t &N, const float *__restrict__ kernel, const int64_t kernel_size, const int64_t repititions, uint8_t *__restrict__ output) {
        const int64_t
            total_size = N.z*N.y*N.x,
            radius = kernel_size / 2;

        float *buf0 = new float[total_size];
        float *buf1 = new float[total_size];

        #pragma acc data copyin(voxels[0:total_size], kernel[0:kernel_size]) copyout(output[0:total_size]) create(buf0[0:total_size], buf1[0:total_size])
        {
            convert_uint8_to_float(voxels, buf0, total_size);
            int64_t total_rep = 0;
            for (int64_t rep = 0; rep < repititions; rep++) {
                for (int64_t dim = 0; dim < 3; dim++) {
                    diffusion_core(total_rep % 2 == 0 ? buf0 : buf1, kernel, total_rep % 2 == 0 ? buf1 : buf0, N, dim, radius, 0);
                    total_rep++;
                }
                illuminate(voxels, total_rep % 2 == 0 ? buf0 : buf1, total_size);
            }
            convert_float_to_uint8(total_rep % 2 == 0 ? buf0 : buf1, output, total_size);
        }

        delete[] buf0;
        delete[] buf1;
    }

    void diffusion_on_disk(const std::string &input_file, const float *__restrict__ kernel, const int64_t kernel_size, const std::string &output_file, const shape_t &total_shape, const shape_t &global_shape, const int64_t repititions, const bool verbose) {
        std::cout << "Not implemented yet" << std::endl;
    }

}