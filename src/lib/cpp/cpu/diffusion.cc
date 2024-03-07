#include "diffusion.hh"

namespace cpu_par {

    void diffusion_core(const float *__restrict__ input, const float *__restrict__ kernel, float *__restrict__ output, const shape_t &N, const int64_t dim, const int64_t radius, const int64_t padding) {
        #pragma omp parallel for collapse(3)
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

                    // Does not add performance:
                    //#pragma omp simd reduction(+:sum)
                    for (int64_t r = ranges[0]; r <= ranges[1]; r++) {
                        const int64_t input_index = output_index + r*stride[dim];
                        float val = input[input_index];
                        sum += val * kernel[radius+r];
                    }

                    output[output_index] = sum;
                }
            }
        }
    }

    void convert_float_to_uint8(const float *__restrict__ src, uint8_t *__restrict__ dst, const int64_t total_flat_size) {
        #pragma omp parallel for
        for (int64_t i = 0; i < total_flat_size; i++) {
            dst[i] = (uint8_t) std::floor(src[i] * 255.0f);
        }
    }

    void convert_uint8_to_float(const uint8_t *__restrict__ src, float *__restrict__ dst, const int64_t total_flat_size) {
        #pragma omp parallel for
        for (int64_t i = 0; i < total_flat_size; i++) {
            dst[i] = src[i] > 0 ? 1.0f : 0.0f;
        }
    }

    void illuminate(const uint8_t *__restrict__ mask, float *__restrict__ output, const int64_t local_flat_size) {
        #pragma omp parallel for
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
        float **buffers = new float*[2];
        buffers[0] = new float[total_size];
        buffers[1] = new float[total_size];

        convert_uint8_to_float(voxels, buffers[0], total_size);
        for (int64_t rep = 0; rep < repititions; rep++) {
            for (int64_t dim = 0; dim < 3; dim++) {
                diffusion_core(buffers[0], kernel, buffers[1], N, dim, radius, 0);
                std::swap(buffers[0], buffers[1]);
            }
            illuminate(voxels, buffers[0], total_size);
        }
        convert_float_to_uint8(buffers[0], output, total_size);

        delete[] buffers[0];
        delete[] buffers[1];
        delete[] buffers;
    }

}