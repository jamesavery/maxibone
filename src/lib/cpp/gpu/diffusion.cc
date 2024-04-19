#include "diffusion.hh"

#include <iostream>
#include <filesystem>

constexpr bool DEBUG = false;

namespace gpu {

    void diffusion_core(const float *__restrict__ input, const float *__restrict__ kernel, float *__restrict__ output, const shape_t &N, const int32_t dim, const int32_t radius) {
        // Note: the use of 32-bit is intentional to reduce register pressure on GPU. Each of the 32-bit values shouldn't exceed 2^32, but the indices to the arrays can be.

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

    void illuminate(const uint8_t *__restrict__ mask, float *__restrict__ output, const int64_t local_flat_size) {
        #pragma acc parallel loop present(mask, output)
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

    void diffusion_step(const uint8_t *__restrict__ voxels, float *buf0, float *buf1, const shape_t &N, const float *__restrict__ kernel, const int64_t radius) {
        for (int64_t dim = 0; dim < 3; dim++) {
            diffusion_core(buf0, kernel, buf1, N, dim, radius);
            std::swap(buf0, buf1);
        }
        illuminate(voxels, buf0, N.z*N.y*N.x);
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
        const int64_t
            total_size = N.z*N.y*N.x,
            radius = kernel_size / 2;

        float *buf0 = new float[total_size];
        float *buf1 = new float[total_size];

        #pragma acc data copyin(voxels[0:total_size], kernel[0:kernel_size]) copyout(output[0:total_size]) create(buf0[0:total_size], buf1[0:total_size])
        {
            convert_uint8_to_float(voxels, buf0, total_size);
            for (int64_t rep = 0; rep < repititions; rep++) {
                diffusion_step(voxels, buf0, buf1, N, kernel, radius);
                std::swap(buf0, buf1);
            }
            convert_float_to_uint16(buf0, output, total_size);
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
                    load_partial(tmp0, buf0, global_flat_size_padded, global_block*global_shape.z*layer_flat_size, this_block_size, total_flat_size, radius*layer_flat_size);

                    #pragma acc data copyin(buf0[:global_flat_size_padded]) copyout(buf1[:global_flat_size_padded])
                    {
                        store_mask(buf0, mask, global_flat_size_padded);
                        diffusion_step(mask, buf0, buf1, global_shape_padded, kernel, radius);
                    }
                    std::swap(buf0, buf1);

                    // Store the global block
                    store_partial(tmp1, buf0, global_block*global_flat_size, this_block_size, radius*layer_flat_size);
                }
                std::swap(temp0, temp1);
                std::swap(tmp0, tmp1);
            }
            std::cout << "\rDiffusion is complete!" << std::endl;
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

}