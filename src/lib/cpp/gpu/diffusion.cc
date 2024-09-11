#include "diffusion.hh"

#include <chrono>
#include <iostream>
#include <filesystem>
#include "openacc.h"
#include "omp.h"

constexpr bool
    DEBUG = false,
    PROFILE = true;

namespace gpu {

    void diffusion_core(const float *__restrict__ input, const float *__restrict__ kernel, float *__restrict__ output, const shape_t &N, const int32_t dim, const int32_t radius, const int stream = 0) {
        // Note: the use of 32-bit is intentional to reduce register pressure on GPU. Each of the 32-bit values shouldn't exceed 2^32, but the indices (address) to the arrays can be.

        const int32_t
            strides[3] = {(N.y)*(N.x), N.x, 1},
            stride = strides[dim],
            Ns[3] = {N.z, N.y, N.x};

        #pragma acc parallel loop collapse(3) present(input, kernel, output) async(stream)
        for (int32_t i = 0; i < N.z; i++) {
            for (int32_t j = 0; j < N.y; j++) {
                for (int32_t k = 0; k < N.x; k++) {
                    const int32_t
                        X[3] = {i, j, k},
                        ranges[2] = {
                            -std::min(radius, X[dim]), std::min(radius, Ns[dim]-X[dim]-1)
                        };
                    const int64_t output_index = (uint64_t)i*(uint64_t)strides[0] + (uint64_t)j*(uint64_t)strides[1] + (uint64_t)k*(uint64_t)strides[2];

                    float sum = 0.0f;

                    for (int32_t r = -radius; r <= radius; r++) {
                        const int64_t input_index = output_index + (int64_t)r*(int64_t)stride;
                        // If the loop ranges guards the accesses
                        //float val = input[input_index];
                        //sum += val * kernel[radius+r];

                        // Branch free version
                        bool cond = r >= ranges[0] && r <= ranges[1];
                        int32_t mask = -cond;
                        mask &= mask >> 31;
                        int64_t index = (mask & input_index) | (~mask & output_index);
                        float val = input[index];

                        // Branchfree version - Multiplication
                        //val *= (float) cond;

                        // Branch free version - pointer casting
                        int32_t *vali = (int32_t*) &val;
                        *vali &= mask;

                        // Branch free version - memcpy
                        //int32_t vali;
                        //std::memcpy(&vali, &val, sizeof(val));
                        //vali &= mask;
                        //std::memcpy(&val, &vali, sizeof(val));

                        // Branch free version - union
                        //union raw32 {
                        //    float f;
                        //    int32_t i;
                        //};
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

    void diffusion_core_og(const float *__restrict__ input, const float *__restrict__ kernel, float *__restrict__ output, const shape_t &N, const int32_t dim, const int32_t radius) {
        // Note: the use of 32-bit is intentional to reduce register pressure on GPU. Each of the 32-bit values shouldn't exceed 2^32, but the indices (address) to the arrays can be.

        const int32_t
            strides[3] = {(N.y)*(N.x), N.x, 1},
            stride = strides[dim],
            Ns[3] = {N.z, N.y, N.x};

        #pragma acc parallel loop collapse(3) present(input, kernel, output)
        for (int32_t i = 0; i < N.z; i++) {
            for (int32_t j = 0; j < N.y; j++) {
                for (int32_t k = 0; k < N.x; k++) {
                    const int32_t
                        X[3] = {i, j, k},
                        ranges[2] = {
                            -std::min(radius, X[dim]), std::min(radius, Ns[dim]-X[dim]-1)
                        };
                    const int64_t output_index = (uint64_t)i*(uint64_t)strides[0] + (uint64_t)j*(uint64_t)strides[1] + (uint64_t)k*(uint64_t)strides[2];

                    float sum = 0.0f;

                    for (int32_t r = -radius; r <= radius; r++) {
                        const int64_t input_index = output_index + (int64_t)r*(int64_t)stride;
                        bool cond = r >= ranges[0] && r <= ranges[1];

                        // Original:
                        float val = cond ? input[input_index] : 0.0f;
                        sum += val * kernel[radius+r];
                    }

                    output[output_index] = sum;
                }
            }
        }
    }

    // TODO Make this call y, since they're VERY similar?
    void diffusion_core_z(const float *__restrict__ input, const float *__restrict__ kernel, float *__restrict__ output, const shape_t &N, const int32_t radius, const int stream = 0) {
        // Assumes that the x dimension is a multiple of veclen.
        constexpr int32_t
            worklen = 1,
            veclen = 32,
            max_k = 32,
            sqvec = max_k*veclen;
        const int32_t
            kernel_size = 2*radius+1,
            nz = N.z, ny = N.y, nx = N.x;
        #pragma acc parallel vector_length(veclen) num_workers(worklen) present(input, kernel, output) async(stream)
        {
            #pragma acc loop gang collapse(2)
            for (int32_t y = 0; y < ny; y++) {
                //#pragma acc loop worker
                for (int32_t x = 0; x < nx; x += veclen) {
                    float local[sqvec], local_kernel[max_k]; // Local memory.
                    #pragma acc cache(local_kernel, local)
                    {
                        #pragma acc loop vector
                        for (int32_t tid = 0; tid < veclen; tid++) {
                            #pragma acc loop seq
                            for (int32_t z = 0; z < radius; z++) {
                                local[z*veclen + tid] = 0; // Zero out the local memory.
                            }
                            #pragma acc loop seq
                            for (int32_t z = radius; z < kernel_size; z++) {
                                local[z*veclen + tid] = input[(z-radius)*ny*nx + y*nx + x + tid];
                            }
                            // Load the kernel into the local memory.
                            local_kernel[tid] = tid < kernel_size ? kernel[tid] : 0;
                        }
                        #pragma acc loop seq
                        for (int32_t z = 0; z < nz; z++) {
                            #pragma acc loop vector
                            for (int32_t tid = 0; tid < veclen; tid++) {
                                float sum = local[tid] * local_kernel[0];
                                #pragma acc loop seq
                                for (int32_t r = 1; r < kernel_size; r++) {
                                    float val = local[r*veclen + tid];
                                    sum += val * local_kernel[r];
                                    local[(r-1)*veclen + tid] = val;
                                }
                                output[z*ny*nx + y*nx + x + tid] = sum;
                                local[(kernel_size-1)*veclen + tid] = z+radius+1 < nz ? input[(z+radius+1)*ny*nx + y*nx + x + tid] : 0;
                            }
                        }
                    }
                }
            }
        }
    }

    void diffusion_core_y(const float *__restrict__ input, const float *__restrict__ kernel, float *__restrict__ output, const shape_t &N, const int32_t radius, const int stream = 0) {
        // Assumes that the x dimension is a multiple of veclen.
        constexpr int32_t
            worklen = 1,
            veclen = 32,
            max_k = 32,
            sqvec = max_k*veclen;
        const int32_t
            kernel_size = 2*radius+1,
            nz = N.z, ny = N.y, nx = N.x;
        #pragma acc parallel vector_length(veclen) num_workers(worklen) present(input, kernel, output) async(stream)
        {
            #pragma acc loop gang collapse(2)
            for (int32_t z = 0; z < nz; z++) {
                //#pragma acc loop worker
                for (int32_t x = 0; x < nx; x += veclen) {
                    float local[sqvec], local_kernel[max_k]; // Local memory.
                    #pragma acc cache(local_kernel, local)
                    {
                        #pragma acc loop vector
                        for (int32_t tid = 0; tid < veclen; tid++) {
                            #pragma acc loop seq
                            for (int32_t y = 0; y < radius; y++) {
                                local[y*veclen + tid] = 0; // Zero out the local memory.
                            }
                            #pragma acc loop seq
                            for (int32_t y = radius; y < kernel_size; y++) {
                                local[y*veclen + tid] = input[z*ny*nx + (y-radius)*nx + x + tid];
                            }
                            // Load the kernel into the local memory.
                            local_kernel[tid] = tid < kernel_size ? kernel[tid] : 0;
                        }
                        #pragma acc loop seq
                        for (int32_t y = 0; y < ny; y++) {
                            #pragma acc loop vector
                            for (int32_t tid = 0; tid < veclen; tid++) {
                                float sum = local[tid] * local_kernel[0];
                                #pragma acc loop seq
                                for (int32_t r = 1; r < kernel_size; r++) {
                                    float val = local[r*veclen + tid];
                                    sum += val * local_kernel[r];
                                    local[(r-1)*veclen + tid] = val;
                                }
                                output[z*ny*nx + y*nx + x + tid] = sum;
                                local[(kernel_size-1)*veclen + tid] = y+radius+1 < ny ? input[z*ny*nx + (y+radius+1)*nx + x + tid] : 0;
                            }
                        }
                    }
                }
            }
        }
    }

    void diffusion_core_x(const float *__restrict__ input, const float *__restrict__ kernel, float *__restrict__ output, const shape_t &N, const int32_t radius, const int stream) {
        // Note: the use of 32-bit is intentional to reduce register pressure on GPU. Each of the 32-bit values shouldn't exceed 2^32, but the indices (address) to the arrays can be.
        constexpr int32_t veclen = 32;
        float local[3*veclen], local_kernel[veclen]; // Local memory.
        const int32_t nz = N.z, ny = N.y, nx = N.x;
        #pragma acc parallel vector_length(veclen) num_workers(1) present(input, kernel, output) async(stream)
        {
            #pragma acc loop gang collapse(2)
            for (int32_t z = 0; z < nz; z++) {
                //#pragma acc loop worker private(local, local_kernel)
                for (int32_t y = 0; y < ny; y++) {
                    #pragma acc cache(local, local_kernel)
                    {
                        { // First iteration
                            int32_t x = 0;
                            #pragma acc loop vector
                            for (int32_t tid = 0; tid < veclen; tid++) {
                                const int64_t input_index = z*ny*nx + y*nx + x + tid;
                                local[0*veclen + tid] = 0;
                                local[1*veclen + tid] = input[input_index];
                                local[2*veclen + tid] = input[input_index + veclen];
                                local_kernel[tid] = tid < (2*radius+1) ? kernel[tid] : 0;
                            }
                            #pragma acc loop vector
                            for (int32_t tid = 0; tid < veclen; tid++) {
                                float sum = 0.0f;
                                #pragma acc loop seq
                                for (int32_t r = -radius; r <= radius; r++) {
                                    sum += local[tid + veclen + r] * local_kernel[r+radius];
                                }
                                output[z*ny*nx + y*nx + x + tid] = sum;
                            }
                        }
                        for (int32_t x = veclen; x < nx-veclen; x += veclen) {
                            #pragma acc loop vector
                            for (int32_t tid = 0; tid < veclen; tid++) {
                                local[0*veclen + tid] = local[1*veclen + tid];
                                local[1*veclen + tid] = local[2*veclen + tid];
                                local[2*veclen + tid] = input[z*ny*nx + y*nx + x + tid + veclen];
                            }
                            #pragma acc loop vector
                            for (int32_t tid = 0; tid < veclen; tid++) {
                                float sum = 0.0f;
                                #pragma acc loop seq
                                for (int32_t r = -radius; r <= radius; r++) {
                                    sum += local[tid + veclen + r] * local_kernel[r+radius];
                                }
                                output[z*ny*nx + y*nx + x + tid] = sum;
                            }
                        }
                        { // Last iteration
                            int32_t x = nx-veclen;
                            #pragma acc loop vector
                            for (int32_t tid = 0; tid < veclen; tid++) {
                                local[0*veclen + tid] = local[1*veclen + tid];
                                local[1*veclen + tid] = local[2*veclen + tid];
                                local[2*veclen + tid] = 0;
                            }
                            #pragma acc loop vector
                            for (int32_t tid = 0; tid < veclen; tid++) {
                                float sum = 0.0f;
                                #pragma acc loop seq
                                for (int32_t r = -radius; r <= radius; r++) {
                                    sum += local[tid + veclen + r] * local_kernel[r+radius];
                                }
                                output[z*ny*nx + y*nx + x + tid] = sum;
                            }
                        }
                    }
                }
            }
        }
    }

    // padding is padding*ny*nx - i.e. number of padding layers in flat size
    template <typename T>
    void load_partial(FILE *f, T *__restrict__ buffer, const int64_t buffer_size, const int64_t offset, const int64_t size, const int64_t total_size, const int64_t padding) {
        const int64_t
            disk_begin = std::max((int64_t) 0, offset-padding),
            disk_end = std::min(offset+size+padding, total_size),
            read_size = disk_end - disk_begin,
            buffer_begin = (offset < padding) ? padding - offset : 0,
            buffer_written = buffer_begin + read_size,
            buffer_remaining = buffer_size - buffer_written;

        assert (buffer_size >= buffer_written + buffer_remaining && "Buffer is too small");
        assert (buffer_remaining >= 0 && "Something went wrong with the buffer size calculation");

        if (DEBUG) {
            std::cout << std::endl;
            std::cout << "buffer_size: " << buffer_size << std::endl;
            std::cout << "offset: " << offset << std::endl;
            std::cout << "size: " << size << std::endl;
            std::cout << "total_size: " << total_size << std::endl;
            std::cout << "padding: " << padding << std::endl;
            std::cout << "disk_begin: " << disk_begin << std::endl;
            std::cout << "disk_end: " << disk_end << std::endl;
            std::cout << "read_size: " << read_size << std::endl;
            std::cout << "buffer_begin: " << buffer_begin << std::endl;
            std::cout << "buffer_written: " << buffer_written << std::endl;
            std::cout << "buffer_remaining: " << buffer_remaining << std::endl;
            std::cout << std::endl;
        }

        // TODO rotate rather than read everything

        // Fill the start of the buffer with zeros
        if (offset < padding) {
            memset(buffer, 0, (padding - offset)*sizeof(T));
        }

        // Read the data
        fseek(f, disk_begin*sizeof(T), SEEK_SET);
        int64_t n = fread(buffer+buffer_begin, sizeof(T), read_size, f);
        assert (n == read_size);

        // Fill the rest of the buffer with zeros
        if (disk_end == total_size) {
            memset(buffer+buffer_written, 0, buffer_remaining*sizeof(T));
        }
    }

    template <typename T>
    void store_partial(FILE *f, const T *__restrict__ buffer, const int64_t offset, const int64_t size, const int64_t padding) {
        fseek(f, offset*sizeof(T), SEEK_SET);
        fwrite(buffer+padding, sizeof(T), size, f);
    }

    FILE* open_file_read(const std::string &path) {
        FILE *file = fopen(path.c_str(), "rb");
        if (file == NULL) {
            throw std::runtime_error("Could not open file: " + path);
        }
        return file;
    }

    FILE* open_file_write(const std::string &path, const int64_t size) {
        FILE *file = fopen(path.c_str(), "wb");
        if (file == NULL) {
            throw std::runtime_error("Could not open file: " + path);
        }
        ftruncate(fileno(file), size);
        return file;
    }

    FILE* open_file_read_write(const std::string &path, const int64_t size) {
        FILE *file = fopen(path.c_str(), "w+b");
        if (file == NULL) {
            throw std::runtime_error("Could not open file: " + path);
        }
        ftruncate(fileno(file), size);
        return file;
    }

    void convert_float_to_uint8(const float *__restrict__ src, uint8_t *__restrict__ dst, const int64_t total_flat_size) {
        #pragma acc parallel loop
        for (int64_t i = 0; i < total_flat_size; i++) {
            dst[i] = (uint8_t) std::floor(src[i] * 255.0f);
        }
    }

    void convert_float_to_uint8(const std::string &src, const std::string &dst, const int64_t total_flat_size) {
        constexpr int64_t
            disk_block_size = 4096,
            chunk_size = 2048*disk_block_size;
        FILE *file_src = open_file_read(src);
        FILE *file_dst = open_file_write(dst, total_flat_size*sizeof(uint8_t));
        float *buffer_src = (float *) aligned_alloc(disk_block_size, chunk_size*sizeof(float));
        uint8_t *buffer_dst = (uint8_t *) aligned_alloc(disk_block_size, chunk_size*sizeof(uint8_t));

        for (int64_t chunk = 0; chunk < total_flat_size; chunk += chunk_size) {
            int64_t size = std::min(chunk_size, total_flat_size - chunk);
            load_partial(file_src, buffer_src, chunk_size, chunk, size, total_flat_size, 0);
            #pragma acc data copyin(buffer_src[0:chunk_size]) create(buffer_dst[0:chunk_size]) copyout(buffer_dst[0:chunk_size])
            {
                convert_float_to_uint8(buffer_src, buffer_dst, size);
            }
            store_partial(file_dst, buffer_dst, chunk, size, 0);
        }

        free(buffer_dst);
        free(buffer_src);
        fclose(file_dst);
        fclose(file_src);
    }

    void convert_float_to_uint16(const float *__restrict__ src, uint16_t *__restrict__ dst, const int64_t total_flat_size) {
        #pragma acc parallel loop
        for (int64_t i = 0; i < total_flat_size; i++) {
            dst[i] = (uint16_t) std::floor(src[i] * 65535.0f);
        }
    }

    void convert_float_to_uint16(const float *__restrict__ src, uint16_t *__restrict__ dst, const shape_t &N, const shape_t &P) {
        #pragma acc parallel loop collapse(3) present(src, dst)
        for (int32_t z = 0; z < N.z; z++) {
            for (int32_t y = 0; y < N.y; y++) {
                for (int32_t x = 0; x < N.x; x++) {
                    const int64_t
                        src_index = (int64_t)z*(int64_t)P.y*(int64_t)P.x + (int64_t)y*(int64_t)P.x + (int64_t)x,
                        dst_index = (int64_t)z*(int64_t)N.y*(int64_t)N.x + (int64_t)y*(int64_t)N.x + (int64_t)x;
                    dst[dst_index] = (uint16_t) std::floor(src[src_index] * 65535.0f);
                }
            }
        }
    }

    void convert_float_to_uint16(const std::string &src, const std::string &dst, const int64_t total_flat_size) {
        constexpr int64_t
            disk_block_size = 4096,
            chunk_size = 2048*disk_block_size;
        FILE *file_src = open_file_read(src);
        FILE *file_dst = open_file_write(dst, total_flat_size*sizeof(uint16_t));
        float *buffer_src = (float *) aligned_alloc(disk_block_size, chunk_size*sizeof(float));
        uint16_t *buffer_dst = (uint16_t *) aligned_alloc(disk_block_size, chunk_size*sizeof(uint16_t));

        for (int64_t chunk = 0; chunk < total_flat_size; chunk += chunk_size) {
            std::cout << "\rConverting: " << chunk / chunk_size << "/" << total_flat_size / chunk_size << std::flush;
            int64_t size = std::min(chunk_size, total_flat_size - chunk);
            load_partial(file_src, buffer_src, chunk_size, chunk, size, total_flat_size, 0);
            #pragma acc data copyin(buffer_src[0:chunk_size]) create(buffer_dst[0:chunk_size]) copyout(buffer_dst[0:chunk_size])
            {
                convert_float_to_uint16(buffer_src, buffer_dst, size);
            }
            store_partial(file_dst, buffer_dst, chunk, size, 0);
        }
        std::cout << "\rConversion is complete!" << std::endl;

        free(buffer_dst);
        free(buffer_src);
        fclose(file_dst);
        fclose(file_src);
    }

    void convert_uint8_to_float(const uint8_t *__restrict__ src, float *__restrict__ dst, const int64_t total_flat_size) {
        #pragma acc parallel loop
        for (int64_t i = 0; i < total_flat_size; i++) {
            dst[i] = src[i] > 0 ? 1.0f : 0.0f;
        }
    }

    void convert_uint8_to_float(const uint8_t *__restrict__ src, float *__restrict__ dst, const shape_t &N, const shape_t &P) {
        #pragma acc parallel loop collapse(3) present(src, dst)
        for (int32_t z = 0; z < (int32_t)P.z; z++) {
            for (int32_t y = 0; y < (int32_t)P.y; y++) {
                for (int32_t x = 0; x < (int32_t)P.x; x++) {
                    const bool
                        valid_z = z < (int32_t)N.z,
                        valid_y = y < (int32_t)N.y,
                        valid_x = x < (int32_t)N.x;
                    const int64_t
                        src_index = (int64_t)z*(int64_t)N.y*(int64_t)N.x + (int64_t)y*(int64_t)N.x + (int64_t)x,
                        dst_index = (int64_t)z*(int64_t)P.y*(int64_t)P.x + (int64_t)y*(int64_t)P.x + (int64_t)x;
                    dst[dst_index] = valid_z && valid_y && valid_x && src[src_index] ? 1.0f : 0.0f;
                }
            }
        }
    }

    void convert_uint8_to_float(const std::string &src, const std::string &dst, const int64_t total_flat_size) {
        constexpr int64_t
            disk_block_size = 4096,
            chunk_size = 2048*disk_block_size;
        FILE *file_src = open_file_read(src);
        FILE *file_dst = open_file_write(dst, total_flat_size*sizeof(float));
        uint8_t *buffer_src = (uint8_t *) aligned_alloc(disk_block_size, chunk_size*sizeof(uint8_t));
        float *buffer_dst = (float *) aligned_alloc(disk_block_size, chunk_size*sizeof(float));

        for (int64_t chunk = 0; chunk < total_flat_size; chunk += chunk_size) {
            std::cout << "\rConverting: " << chunk / chunk_size << "/" << total_flat_size / chunk_size << std::flush;
            int64_t size = std::min(chunk_size, total_flat_size - chunk);
            load_partial(file_src, buffer_src, chunk_size, chunk, size, total_flat_size, 0);
            #pragma acc data copyin(buffer_src[0:chunk_size]) create(buffer_dst[0:chunk_size]) copyout(buffer_dst[0:chunk_size])
            {
                convert_uint8_to_float(buffer_src, buffer_dst, size);
            }
            store_partial(file_dst, buffer_dst, chunk, size, 0);
        }
        std::cout << "\rConversion is complete!" << std::endl;

        free(buffer_dst);
        free(buffer_src);
        fclose(file_dst);
        fclose(file_src);
    }

    void illuminate(const uint8_t *__restrict__ mask, float *__restrict__ output, const int64_t local_flat_size, const int stream = 0) {
        #pragma acc parallel loop present(mask, output) async(stream)
        for (int64_t thread = 0; thread < gpu_threads; thread++) {
            const int64_t
                chunk_size = local_flat_size / gpu_threads,
                start_x = thread * chunk_size,
                end_x = std::min(local_flat_size, (thread+1) * chunk_size);
            for (int64_t i = start_x; i < end_x; i++) {
                output[i] = mask[i] ? 1.0f : output[i];
            }
        }
    }

    void illuminate(const uint8_t *__restrict__ mask, float *__restrict__ output, const shape_t &N, const shape_t &P, const int stream = 0) {
        #pragma acc parallel loop collapse(3) present(mask, output) async(stream)
        for (int32_t z = 0; z < N.z; z++) {
            for (int32_t y = 0; y < N.y; y++) {
                for (int32_t x = 0; x < N.x; x++) {
                    const int64_t
                        src_index = (int64_t)z*(int64_t)N.y*(int64_t)N.x + (int64_t)y*(int64_t)N.x + (int64_t)x,
                        dst_index = (int64_t)z*(int64_t)P.y*(int64_t)P.x + (int64_t)y*(int64_t)P.x + (int64_t)x;
                    output[dst_index] = mask[src_index] ? 1.0f : output[dst_index];
                }
            }
        }
    }

    void diffusion_step(const uint8_t *__restrict__ voxels, float *buf0, float *buf1, const shape_t &N, const shape_t &P, const float *__restrict__ kernel, const int64_t radius, const int stream = 0) {
        if (radius < 16) {
            diffusion_core_z(buf0, kernel, buf1, P, radius, stream);
            diffusion_core_y(buf1, kernel, buf0, P, radius, stream);
            diffusion_core_x(buf0, kernel, buf1, P, radius, stream);
        } else {
            diffusion_core(buf0, kernel, buf1, P, 0, radius, stream);
            diffusion_core(buf1, kernel, buf0, P, 1, radius, stream);
            diffusion_core(buf0, kernel, buf1, P, 2, radius, stream);
        }
        std::swap(buf0, buf1);

        illuminate(voxels, buf0, N, P, stream);
    }

    void store_mask(const float *__restrict__ input, uint8_t *__restrict__ mask, const int64_t local_flat_size) {
        #pragma acc parallel loop present(input, mask)
        for (int64_t thread = 0; thread < gpu_threads; thread++) {
            const int64_t
                chunk_size = local_flat_size / gpu_threads,
                start_x = thread * chunk_size,
                end_x = std::min(local_flat_size, (thread+1) * chunk_size);
            for (int64_t i = start_x; i < end_x; i++) {
                mask[i] = (input[i] == 1.0f); // The implant will always be 1.0f
            }
        }
    }

    void diffusion_in_memory(const uint8_t *__restrict__ voxels, const shape_t &N, const float *__restrict__ kernel, const int64_t kernel_size, const int64_t repititions, uint16_t *__restrict__ output) {
        constexpr int32_t veclen = 32; // TODO
        const shape_t P = {
            ((N.z + veclen-1) / veclen) * veclen,
            ((N.y + veclen-1) / veclen) * veclen,
            ((N.x + veclen-1) / veclen) * veclen
        };
        const int64_t
            padded_size = P.z*P.y*P.x,
            total_size = N.z*N.y*N.x,
            radius = kernel_size / 2;

        float *buf0 = new float[padded_size];
        float *buf1 = new float[padded_size];

        #pragma acc data copyin(voxels[0:total_size], kernel[0:kernel_size]) copyout(output[0:total_size]) create(buf0[0:padded_size], buf1[0:padded_size])
        {
            convert_uint8_to_float(voxels, buf0, N, P);
            for (int64_t rep = 0; rep < repititions; rep++) {
                diffusion_step(voxels, buf0, buf1, N, P, kernel, radius);
                std::swap(buf0, buf1);
            }
            convert_float_to_uint16(buf0, output, N, P);
        }

        delete[] buf0;
        delete[] buf1;
    }

    void diffusion_on_disk(const std::string &input_file, const float *__restrict__ kernel, const int64_t kernel_size, const std::string &output_file, const shape_t &total_shape, const shape_t &global_shape, const int64_t repititions, const bool verbose) {
        std::string
            temp_folder = "/tmp/maxibone",
            temp0 = temp_folder + "/diffusion-temp0.float32",
            temp1 = temp_folder + "/diffusion-temp1.float32";

        // Create temp folder
        std::filesystem::create_directories(temp_folder);

        // Compute the number of global blocks
        const int64_t
            radius = kernel_size / 2,
            padding = kernel_size - 1, // kernel is assumed to be odd
            disk_block_size = 4096, // TODO Generate a header file during configuration
            total_flat_size = total_shape.z * total_shape.y * total_shape.x,
            global_blocks = (int64_t) std::ceil((float)total_shape.z / (float)global_shape.z),
            global_flat_size = global_shape.z * global_shape.y * global_shape.x,
            global_flat_size_padded = (global_shape.z+padding) * global_shape.y * global_shape.x,
            layer_flat_size = global_shape.y * global_shape.x;

        const shape_t global_shape_padded = {global_shape.z+padding, global_shape.y, global_shape.x};

        if (DEBUG) {
            std::cout << "Radius: " << radius << std::endl;
            std::cout << "Padding: " << padding << std::endl;
            std::cout << "Disk block size: " << disk_block_size << std::endl;
            std::cout << "Total flat size: " << total_flat_size << std::endl;
            std::cout << "Global blocks: " << global_blocks << std::endl;
            std::cout << "Global flat size: " << global_flat_size << std::endl;
            std::cout << "Global flat size padded: " << global_flat_size_padded << std::endl;
            std::cout << "Layer flat size: " << layer_flat_size << std::endl;
        }

        // Allocate memory. Aligned to block_size, and should overallocate to ensure alignment.
        // TODO since the I/O functions handle alignment with buffers, the allocations doesn't need to be aligned. Although, it might make sense to do so anyways, since this can avoid the need for a buffer. However, checking this is complicated, and is left for later.
        float
            *buf0 = (float *) malloc((global_flat_size_padded) * sizeof(float)),
            *buf1 = (float *) malloc((global_flat_size_padded) * sizeof(float));
            //*buf0 = (float *) aligned_alloc(disk_block_size, disk_global_padded_flat_size),
            //*buf1 = (float *) aligned_alloc(disk_block_size, disk_global_padded_flat_size);
        uint8_t *mask = (uint8_t *) malloc((global_flat_size_padded) * sizeof(uint8_t));

        FILE
            *tmp0 = open_file_read_write(temp0, total_flat_size),
            *tmp1 = open_file_read_write(temp1, total_flat_size);

        double *loads, *diffusions, *stores;
        uint64_t *sizes;
        if (PROFILE) {
            std::cout << "Profiling enabled" << std::endl;
            loads = (double *) malloc(repititions * global_blocks * sizeof(double));
            diffusions = (double *) malloc(repititions * global_blocks * sizeof(double));
            stores = (double *) malloc(repititions * global_blocks * sizeof(double));
            sizes = (uint64_t *) malloc(repititions * global_blocks * sizeof(uint64_t));
        }

        // Convert to float
        convert_uint8_to_float(input_file, temp0, total_flat_size);

        #pragma acc data copyin(kernel[:kernel_size]) create(mask[:global_flat_size_padded])
        {
            for (int64_t rep = 0; rep < repititions; rep++) {
                for (int64_t global_block = 0; global_block < global_blocks; global_block++) {
                    std::cout << "\rDiffusion: " << rep*global_blocks + global_block << "/" << repititions*global_blocks << std::flush;
                    int64_t this_block_size = std::min(global_shape.z, total_shape.z - global_block*global_shape.z) * layer_flat_size;
                    assert (this_block_size <= global_flat_size_padded);

                    // Load the global block
                    auto load_start = std::chrono::high_resolution_clock::now();
                    load_partial(tmp0, buf0, global_flat_size_padded, global_block*global_shape.z*layer_flat_size, this_block_size, total_flat_size, radius*layer_flat_size);
                    auto load_end = std::chrono::high_resolution_clock::now();

                    auto diffusion_start = std::chrono::high_resolution_clock::now();
                    #pragma acc data copyin(buf0[:global_flat_size_padded]) copyout(buf1[:global_flat_size_padded])
                    {
                        store_mask(buf0, mask, global_flat_size_padded);
                        diffusion_step(mask, buf0, buf1, global_shape_padded, global_shape_padded, kernel, radius);
                    }
                    std::swap(buf0, buf1);
                    auto diffusion_end = std::chrono::high_resolution_clock::now();

                    auto store_start = std::chrono::high_resolution_clock::now();
                    // Store the global block
                    store_partial(tmp1, buf0, global_block*global_flat_size, this_block_size, radius*layer_flat_size);
                    auto store_end = std::chrono::high_resolution_clock::now();

                    if (PROFILE) {
                        loads[rep*global_blocks + global_block] = std::chrono::duration_cast<std::chrono::nanoseconds>(load_end - load_start).count() / 1e9;
                        diffusions[rep*global_blocks + global_block] = std::chrono::duration_cast<std::chrono::nanoseconds>(diffusion_end - diffusion_start).count() / 1e9;
                        stores[rep*global_blocks + global_block] = std::chrono::duration_cast<std::chrono::nanoseconds>(store_end - store_start).count() / 1e9;
                        sizes[rep*global_blocks + global_block] = this_block_size;
                    }
                }
                std::swap(temp0, temp1);
                std::swap(tmp0, tmp1);
            }
            std::cout << "\rDiffusion is complete!" << std::endl;

            if (PROFILE) {
                double
                    mean_load = 0.0, mean_diffusion = 0.0, mean_store = 0.0,
                    mean_load_gbps = 0.0, mean_store_gbps = 0.0;
                for (int64_t i = 0; i < repititions*global_blocks; i++) {
                    double
                        bytes = (double) (sizes[i] * sizeof(float)),
                        gigabytes = bytes / (double)1e9;
                    mean_load += loads[i];
                    mean_diffusion += diffusions[i];
                    mean_store += stores[i];
                    mean_load_gbps += gigabytes / loads[i];
                    mean_store_gbps += gigabytes / stores[i];
                }
                mean_load /= (double)(repititions*global_blocks);
                mean_diffusion /= (double)(repititions*global_blocks);
                mean_store /= (double)(repititions*global_blocks);
                mean_load_gbps /= (double)(repititions*global_blocks);
                mean_store_gbps /= (double)(repititions*global_blocks);

                std::cout << "Mean load: " << mean_load << " (" << mean_load_gbps << " GB/s)" << std::endl;
                std::cout << "Mean diffusion: " << mean_diffusion << std::endl;
                std::cout << "Mean store: " << mean_store << " (" << mean_store_gbps << " GB/s)" << std::endl;
            }
        }

        // Convert to uint16
        convert_float_to_uint16(temp0, output_file, total_flat_size);

        // Free memory
        free(buf0);
        free(buf1);
        free(mask);
        fclose(tmp0);
        fclose(tmp1);
    }

    // The full data sample resides in main memory, but out of GPU memory.
    // Assumes that each shape in global_shape is > kernel_size in order to contain the overlap. This assumption allows for controlling the memory used by each thread / async operation outside of this call.
    void diffusion_out_of_core(uint8_t *__restrict__ voxels, const shape_t &total_shape, const shape_t &global_shape, const float *__restrict__ kernel, const int64_t kernel_size, const int64_t repititions, uint16_t *__restrict__ output) {
        // TODO should be a configuration parameter set somewhere global / statically during configuration/compilation.
        const int32_t
            veclen = 32,
            #pragma diag_suppress 177 // Ignore the n_devices unused warning - it is used in the pragma below.
            n_devices = acc_get_num_devices(acc_device_nvidia),
            n_streams = 8;
        const int64_t radius = kernel_size / 2;
        const shape_t
            // TODO assume that global_shape is also aligned to veclen?
            global_shape_padded = {
                (((global_shape.z + 2*radius) + veclen - 1) / veclen) * veclen,
                (((global_shape.y + 2*radius) + veclen - 1) / veclen) * veclen,
                (((global_shape.x + 2*radius) + veclen - 1) / veclen) * veclen
            },
            blocks_shape = {
                (total_shape.z + global_shape.z - 1) / global_shape.z,
                (total_shape.y + global_shape.y - 1) / global_shape.y,
                (total_shape.x + global_shape.x - 1) / global_shape.x
            };
        const int64_t
            total_size = total_shape.z*total_shape.y*total_shape.x,
            global_size_padded = global_shape_padded.z*global_shape_padded.y*global_shape_padded.x;

        float *buf0 = (float *) malloc(total_size * sizeof(float));
        float *buf1 = (float *) malloc(total_size * sizeof(float));

        // Should be faster on CPU, compared to GPU, since the data has to come back to the CPU anyways due to the size. For in-memory, the GPU version is faster, since the transfer back can be avoided.
        // This should be tested to confirm.
        #pragma omp parallel for collapse(3) schedule(static)
        for (int32_t z = 0; z < total_shape.z; z++) {
            for (int32_t y = 0; y < total_shape.y; y++) {
                for (int32_t x = 0; x < total_shape.x; x++) {
                    bool
                        z_valid = z < total_shape.z,
                        y_valid = y < total_shape.y,
                        x_valid = x < total_shape.x;
                    const int64_t
                        src_index = (int64_t)z*(int64_t)total_shape.y*(int64_t)total_shape.x + (int64_t)y*(int64_t)total_shape.x + (int64_t)x,
                        dst_index = (int64_t)z*(int64_t)total_shape.y*(int64_t)total_shape.x + (int64_t)y*(int64_t)total_shape.x + (int64_t)x;
                    buf0[dst_index] = z_valid && y_valid && x_valid && voxels[src_index] ? 1.0f : 0.0f;
                }
            }
        }

        #pragma omp parallel num_threads(n_devices * n_streams)
        {
            int32_t
                tid = omp_get_thread_num(),
                __attribute__((unused)) device = tid / n_streams,
                stream = tid % n_streams;
            acc_set_device_num(device, acc_device_nvidia);

            uint8_t *voxels_stage = (uint8_t *) malloc(global_size_padded * sizeof(uint8_t));
            float *buf0_stage = (float *) malloc(global_size_padded * sizeof(float));
            float *buf1_stage = (float *) malloc(global_size_padded * sizeof(float));

            #pragma acc data create(voxels_stage[:global_size_padded], buf0_stage[:global_size_padded], buf1_stage[:global_size_padded]) copyin(kernel[:kernel_size])
            {
            for (int64_t rep = 0; rep < repititions; rep++) {
                #pragma omp for collapse(3) schedule(dynamic)
                for (int32_t bz = 0; bz < blocks_shape.z; bz++) {
                    for (int32_t by = 0; by < blocks_shape.y; by++) {
                        for (int32_t bx = 0; bx < blocks_shape.x; bx++) {
                            const int64_t
                                start_z = bz * global_shape.z,
                                start_y = by * global_shape.y,
                                start_x = bx * global_shape.x,
                                end_z = std::min(total_shape.z, start_z + global_shape.z),
                                end_y = std::min(total_shape.y, start_y + global_shape.y),
                                end_x = std::min(total_shape.x, start_x + global_shape.x);

                            // Copy the data to the staging area
                            for (int32_t z = 0; z < global_shape_padded.z; z++) {
                                for (int32_t y = 0; y < global_shape_padded.y; y++) {
                                    for (int32_t x = 0; x < global_shape_padded.x; x++) {
                                        const int64_t
                                            offsetted_z = start_z + z - radius,
                                            offsetted_y = start_y + y - radius,
                                            offsetted_x = start_x + x - radius,
                                            src_index = (int64_t)offsetted_z*(int64_t)total_shape.y*(int64_t)total_shape.x + (int64_t)offsetted_y*(int64_t)total_shape.x + (int64_t)offsetted_x,
                                            dst_index = (int64_t)z*(int64_t)global_shape_padded.y*(int64_t)global_shape_padded.x + (int64_t)y*(int64_t)global_shape_padded.x + (int64_t)x;
                                        const bool
                                            valid_z = offsetted_z >= 0 && offsetted_z < total_shape.z,
                                            valid_y = offsetted_y >= 0 && offsetted_y < total_shape.y,
                                            valid_x = offsetted_x >= 0 && offsetted_x < total_shape.x;
                                        voxels_stage[dst_index] = valid_z && valid_y && valid_x ? voxels[src_index] : 0;
                                        buf0_stage[dst_index] = valid_z && valid_y && valid_x ? buf0[src_index] : 0;
                                    }
                                }
                            }

                            #pragma acc update device(voxels_stage[:global_size_padded], buf0_stage[:global_size_padded])
                            diffusion_step(voxels_stage, buf0_stage, buf1_stage, global_shape_padded, global_shape_padded, kernel, radius);
                            #pragma acc update self(buf1_stage[:global_size_padded])

                            // Copy the data back from the staging area
                            for (int32_t z = 0; z < global_shape.z; z++) {
                                for (int32_t y = 0; y < global_shape.y; y++) {
                                    for (int32_t x = 0; x < global_shape.x; x++) {
                                        const int64_t
                                            src_index = ((int64_t)z+radius)*(int64_t)global_shape_padded.y*(int64_t)global_shape_padded.x + ((int64_t)y+radius)*(int64_t)global_shape_padded.x + (int64_t)x+radius,
                                            dst_index = ((int64_t)start_z+(int64_t)z)*(int64_t)total_shape.y*(int64_t)total_shape.x + ((int64_t)start_y+(int64_t)y)*(int64_t)total_shape.x + (int64_t)start_x+(int64_t)x;
                                        buf1[dst_index] = buf1_stage[src_index];
                                    }
                                }
                            }
                        }
                    }
                }
                std::swap(buf0, buf1);
            }
            }

            free(voxels_stage);
            free(buf0_stage);
            free(buf1_stage);
        }

        // Same argument as above - should be faster for CPU.
        #pragma omp parallel for collapse(3) schedule(static)
        for (int32_t z = 0; z < total_shape.z; z++) {
            for (int32_t y = 0; y < total_shape.y; y++) {
                for (int32_t x = 0; x < total_shape.x; x++) {
                    const int64_t
                        src_index = (int64_t)z*(int64_t)total_shape.y*(int64_t)total_shape.x + (int64_t)y*(int64_t)total_shape.x + (int64_t)x,
                        dst_index = (int64_t)z*(int64_t)total_shape.y*(int64_t)total_shape.x + (int64_t)y*(int64_t)total_shape.x + (int64_t)x;
                    output[dst_index] = (uint16_t) std::floor(buf0[src_index] * 65535.0f);
                }
            }
        }

        free(buf0);
        free(buf1);
    }

}