#include "diffusion.hh"

#include <fcntl.h>
#include <iostream>
#include <unistd.h>
#include <filesystem>

namespace cpu_par {

    void diffusion_core(const float *__restrict__ input, const float *__restrict__ kernel, float *__restrict__ output, const shape_t &N, const int64_t dim, const int64_t radius) {
        #pragma omp parallel for collapse(3)
        for (int64_t i = 0; i < N.z; i++) {
            for (int64_t j = 0; j < N.y; j++) {
                for (int64_t k = 0; k < N.x; k++) {
                    const int64_t
                        X[3] = {i, j, k},
                        stride[3] = {N.y*N.x, N.x, 1},
                        Ns[3] = {N.z, N.y, N.x},
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

    // padding is padding*ny*nx - i.e. number of padding layers in flat size
    template <typename T>
    void load_partial(FILE *f, T *__restrict__ buffer, const int64_t offset, const int64_t size, const int64_t total_size, const int64_t padding) {
        int64_t
            disk_begin = std::max((int64_t) 0, offset-padding),
            disk_end = std::min(offset+size+padding, total_size),
            disk_size = disk_end - disk_begin,
            buffer_begin = disk_begin < padding ? 0 : padding;

        fseek(f, disk_begin*sizeof(T), SEEK_SET);
        fread(buffer+buffer_begin, sizeof(T), disk_size, f);
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
        #pragma omp parallel for
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
            load_partial(file_src, buffer_src, chunk, size, total_flat_size, 0);
            convert_float_to_uint8(buffer_src, buffer_dst, size);
            store_partial(file_dst, buffer_dst, chunk, size, 0);
        }

        free(buffer_dst);
        free(buffer_src);
        fclose(file_dst);
        fclose(file_src);
    }

    void convert_uint8_to_float(const uint8_t *__restrict__ src, float *__restrict__ dst, const int64_t total_flat_size) {
        #pragma omp parallel for
        for (int64_t i = 0; i < total_flat_size; i++) {
            dst[i] = src[i] > 0 ? 1.0f : 0.0f;
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
            int64_t size = std::min(chunk_size, total_flat_size - chunk);
            load_partial(file_src, buffer_src, chunk, size, total_flat_size, 0);
            convert_uint8_to_float(buffer_src, buffer_dst, size);
            store_partial(file_dst, buffer_dst, chunk, size, 0);
        }

        free(buffer_dst);
        free(buffer_src);
        fclose(file_dst);
        fclose(file_src);
    }

    void illuminate(const uint8_t *__restrict__ mask, float *__restrict__ output, const int64_t local_flat_size) {
        #pragma omp parallel for
        for (int64_t i = 0; i < local_flat_size; i++) {
            if (mask[i] > 0) {
                output[i] = 1.0f;
            }
        }
    }

    void diffusion_step(const uint8_t *__restrict__ voxels, float **buffers, const shape_t &N, const float *__restrict__ kernel, const int64_t radius) {
        for (int64_t dim = 0; dim < 3; dim++) {
            diffusion_core(buffers[0], kernel, buffers[1], N, dim, radius);
            std::swap(buffers[0], buffers[1]);
        }
        illuminate(voxels, buffers[0], N.z*N.y*N.x);
    }

    void store_mask(const float *__restrict__ input, uint8_t *__restrict__ mask, const int64_t local_flat_size) {
        #pragma omp parallel for
        for (int64_t i = 0; i < local_flat_size; i++) {
            mask[i] = input[i] == 1.0f ? 1 : 0; // The implant will always be 1.0f
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
            diffusion_step(voxels, buffers, N, kernel, radius);
        }
        convert_float_to_uint8(buffers[0], output, total_size);

        delete[] buffers[0];
        delete[] buffers[1];
        delete[] buffers;
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
            layer_flat_size = global_shape.y * global_shape.x,
            disk_total_flat_size = ((total_flat_size*sizeof(float) / disk_block_size) + (total_flat_size*sizeof(float) % disk_block_size == 0 ? 0 : 1)) * disk_block_size,
            disk_global_flat_size = ((global_flat_size_padded*sizeof(float) / disk_block_size) + (global_flat_size_padded*sizeof(float) % disk_block_size == 0 ? 0 : 1)) * disk_block_size,
            disk_mask_flat_size = ((global_flat_size_padded*sizeof(bool) / disk_block_size) + (global_flat_size_padded*sizeof(bool) % disk_block_size == 0 ? 0 : 1)) * disk_block_size;

        const shape_t global_shape_padded = {global_shape.z+padding, global_shape.y, global_shape.x};

        if (verbose) {
            // Print the number of blocks
            std::cout << "Global blocks: " << global_blocks << std::endl;
        }

        // Allocate memory. Aligned to block_size, and should overallocate to ensure alignment.
        // TODO since the I/O functions handle alignment with buffers, the allocations doesn't need to be aligned. Although, it might make sense to do so anyways, since this can avoid the need for a buffer. However, checking this is complicated, and is left for later.
        float **buffers = new float*[2];
        buffers[0] = (float *) aligned_alloc(disk_block_size, disk_global_flat_size);
        buffers[1] = (float *) aligned_alloc(disk_block_size, disk_global_flat_size);
        uint8_t *mask = (uint8_t *) aligned_alloc(disk_block_size, disk_mask_flat_size);

        FILE
            *tmp0 = open_file_read_write(temp0, disk_total_flat_size),
            *tmp1 = open_file_read_write(temp1, disk_total_flat_size);

        // Convert to float
        convert_uint8_to_float(input_file, temp0, total_flat_size);

        for (int64_t rep = 0; rep < repititions; rep++) {
            for (int64_t global_block = 0; global_block < global_blocks; global_block++) {
                int64_t this_block_size = std::min(global_shape.z, total_shape.z - global_block*global_shape.z) * layer_flat_size;

                // Load the global block
                load_partial(tmp0, buffers[0], global_block*layer_flat_size, this_block_size, total_flat_size, padding*layer_flat_size);

                // Diffuse the global block
                store_mask(buffers[0], mask, global_flat_size_padded);
                diffusion_step(mask, buffers, global_shape_padded, kernel, radius);

                // Store the global block
                store_partial(tmp1, buffers[0], global_block*global_flat_size, this_block_size, padding*layer_flat_size);
            }
            std::swap(temp0, temp1);
            std::swap(tmp0, tmp1);
        }

        // Convert to uint8
        convert_float_to_uint8(output_file, temp0, total_flat_size);

        // Free memory
        free(buffers[0]);
        free(buffers[1]);
        delete[] buffers;
        free(mask);
        fclose(tmp0);
        fclose(tmp1);
    }

}