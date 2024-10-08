/**
 * @file diffusion.cc
 * @author Carl-Johannes Johnsen (carl-johannes@di.ku.dk)
 * @brief Parallel CPU implementation of the diffusion approximation algorithm.
 * @version 0.1
 * @date 2024-09-16
 *
 * @copyright Copyright (c) 2024
 */
#include "diffusion.hh"

#include <fcntl.h>
#include <iostream>
#include <unistd.h>
#include <filesystem>

namespace cpu_par {

    /**
     * Diffusion core function. This function is the core of the diffusion algorithm, and is called for each dimension.
     * It convolves `kernel` alongside the dimension specified by `dim`.
     * It does not handle padding, and out-of-bound voxels are treated as zero - i.e. they are ignored.
     *
     * @param input The input volume.
     * @param kernel The kernel to convolve with.
     * @param output The output volume.
     * @param N The shape of the volume.
     * @param dim The dimension to convolve along.
     * @param radius The radius of the kernel.
     */
    void diffusion_core(const float *__restrict__ input, const float *__restrict__ kernel, float *__restrict__ output, const shape_t &N, const int64_t dim, const int64_t radius) {
        // TODO Transposed write-back should be faster, as each the reads will be sequential for all three dimensions. Test it!
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

    // TODO Use I/O functions rather than reimplementing the same functionality!
    // padding is padding*ny*nx - i.e. number of padding layers in flat size
    template <typename T>
    void load_partial(FILE *f, T *__restrict__ buffer, const int64_t buffer_size, const int64_t offset, const int64_t size, const int64_t total_size, const int64_t padding) {
        int64_t
            disk_begin = std::max((int64_t) 0, offset-padding),
            disk_end = std::min(offset+size+padding, total_size),
            read_size = disk_end - disk_begin,
            buffer_begin = disk_begin == 0 ? padding : 0,
            buffer_written = buffer_begin + read_size,
            buffer_remaining = buffer_size - buffer_written;

        // Fill the start of the buffer with zeros
        if (disk_begin == 0) {
            memset(buffer, 0, padding*sizeof(T));
        }

        // Read the data
        fseek(f, disk_begin*sizeof(T), SEEK_SET);
        auto n = fread(buffer+buffer_begin, sizeof(T), read_size, f);

        if (n != (uint64_t) read_size) {
            throw std::runtime_error("Could not read the entire buffer");
        }

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
        auto ret = ftruncate(fileno(file), size);
        if (ret != 0) {
            throw std::runtime_error("Could not truncate file: " + path);
        }
        return file;
    }

    FILE* open_file_read_write(const std::string &path, const int64_t size) {
        FILE *file = fopen(path.c_str(), "w+b");
        if (file == NULL) {
            throw std::runtime_error("Could not open file: " + path);
        }
        auto ret = ftruncate(fileno(file), size);
        if (ret != 0) {
            throw std::runtime_error("Could not truncate file: " + path);
        }
        return file;
    }

    /**
     * Converts `src` from `float` to `uint8_t` and stores the result in `dst`.
     * The arrays are assumed to be of the same size.
     *
     * @param src The input array.
     * @param dst The output array.
     * @param total_flat_size The size of the array.
     */
    void convert_float_to_uint8(const float *__restrict__ src, uint8_t *__restrict__ dst, const int64_t total_flat_size) {
        #pragma omp parallel for
        for (int64_t i = 0; i < total_flat_size; i++) {
            dst[i] = (uint8_t) std::floor(src[i] * 255.0f);
        }
    }

    /**
     * Converts the contents of a file `src` from `float` to `uint8_t` and stores the result in a new file `dst`.
     * The files are assumed to have the same size.
     *
     * @param src The path to the input file.
     * @param dst The path to the output file.
     * @param total_flat_size The size of each file.
     * @param verbose The verbosity level. Default is 0.
     */
    void convert_float_to_uint8(const std::string &src, const std::string &dst, const int64_t total_flat_size, const int verbose = 0) {
        constexpr int64_t
            disk_block_size = 4096,
            chunk_size = 2048*disk_block_size;
        FILE *file_src = open_file_read(src);
        FILE *file_dst = open_file_write(dst, total_flat_size*sizeof(uint8_t));
        float *buffer_src = (float *) aligned_alloc(disk_block_size, chunk_size*sizeof(float));
        uint8_t *buffer_dst = (uint8_t *) aligned_alloc(disk_block_size, chunk_size*sizeof(uint8_t));

        for (int64_t chunk = 0; chunk < total_flat_size; chunk += chunk_size) {
            if (verbose >= 1) {
                std::cout << "\rConverting float to uint8: " << chunk / chunk_size << "/" << total_flat_size / chunk_size << std::flush;
            }
            int64_t size = std::min(chunk_size, total_flat_size - chunk);
            load_partial(file_src, buffer_src, chunk_size, chunk, size, total_flat_size, 0);
            convert_float_to_uint8(buffer_src, buffer_dst, size);
            store_partial(file_dst, buffer_dst, chunk, size, 0);
        }
        if (verbose >= 1) {
            std::cout << "\rConversion of float to uint8 is complete!" << std::endl;
        }

        free(buffer_dst);
        free(buffer_src);
        fclose(file_dst);
        fclose(file_src);
    }

    /**
     * Converts `src` from `float` to `uint16_t` and stores the result in `dst`.
     * The arrays are assumed to be of the same size.
     *
     * @param src The input array.
     * @param dst The output array.
     * @param total_flat_size The size of the array.
     */
    void convert_float_to_uint16(const float *__restrict__ src, uint16_t *__restrict__ dst, const int64_t total_flat_size) {
        #pragma omp parallel for
        for (int64_t i = 0; i < total_flat_size; i++) {
            dst[i] = (uint16_t) std::floor(src[i] * 65535.0f);
        }
    }

    /**
     * Converts the contents of a file `src` from `float` to `uint16_t` and stores the result in a new file `dst`.
     * The files are assumed to have the same size.
     *
     * @param src The path to the input file.
     * @param dst The path to the output file.
     * @param total_flat_size The size of each file.
     * @param verbose The verbosity level. Default is 0.
     */
    void convert_float_to_uint16(const std::string &src, const std::string &dst, const int64_t total_flat_size, const int verbose = 0) {
        constexpr int64_t
            disk_block_size = 4096,
            chunk_size = 2048*disk_block_size;
        FILE *file_src = open_file_read(src);
        FILE *file_dst = open_file_write(dst, total_flat_size*sizeof(uint16_t));
        float *buffer_src = (float *) aligned_alloc(disk_block_size, chunk_size*sizeof(float));
        uint16_t *buffer_dst = (uint16_t *) aligned_alloc(disk_block_size, chunk_size*sizeof(uint16_t));

        for (int64_t chunk = 0; chunk < total_flat_size; chunk += chunk_size) {
            if (verbose >= 1) {
                std::cout << "\rConverting float to uint16: " << chunk / chunk_size << "/" << total_flat_size / chunk_size << std::flush;
            }
            int64_t size = std::min(chunk_size, total_flat_size - chunk);
            load_partial(file_src, buffer_src, chunk_size, chunk, size, total_flat_size, 0);
            convert_float_to_uint16(buffer_src, buffer_dst, size);
            store_partial(file_dst, buffer_dst, chunk, size, 0);
        }
        if (verbose >= 1) {
            std::cout << "\rConversion of float to uint16 is complete!" << std::endl;
        }

        free(buffer_dst);
        free(buffer_src);
        fclose(file_dst);
        fclose(file_src);
    }

    /**
     * Converts `src` from `uint8_t` to `float` and stores the result in `dst`.
     * The arrays are assumed to be of the same size.
     *
     * @param src The input array.
     * @param dst The output array.
     * @param total_flat_size The size of the array.
     */
    void convert_uint8_to_float(const uint8_t *__restrict__ src, float *__restrict__ dst, const int64_t total_flat_size) {
        #pragma omp parallel for
        for (int64_t i = 0; i < total_flat_size; i++) {
            dst[i] = src[i] > 0 ? 1.0f : 0.0f;
        }
    }

    /**
     * Converts the contents of a file `src` from `uint8_t` to `float` and stores the result in a new file `dst`.
     * The files are assumed to have the same size.
     *
     * @param src The path to the input file.
     * @param dst The path to the output file.
     * @param total_flat_size The size of each file.
     * @param verbose The verbosity level. Default is 0.
     */
    void convert_uint8_to_float(const std::string &src, const std::string &dst, const int64_t total_flat_size, const int verbose = 0) {
        constexpr int64_t
            disk_block_size = 4096,
            chunk_size = 2048*disk_block_size;
        FILE *file_src = open_file_read(src);
        FILE *file_dst = open_file_write(dst, total_flat_size*sizeof(float));
        uint8_t *buffer_src = (uint8_t *) aligned_alloc(disk_block_size, chunk_size*sizeof(uint8_t));
        float *buffer_dst = (float *) aligned_alloc(disk_block_size, chunk_size*sizeof(float));

        for (int64_t chunk = 0; chunk < total_flat_size; chunk += chunk_size) {
            if (verbose >= 1) {
                std::cout << "\rConverting uint8 to float: " << chunk / chunk_size << "/" << total_flat_size / chunk_size << std::flush;
            }
            int64_t size = std::min(chunk_size, total_flat_size - chunk);
            load_partial(file_src, buffer_src, chunk_size, chunk, size, total_flat_size, 0);
            convert_uint8_to_float(buffer_src, buffer_dst, size);
            store_partial(file_dst, buffer_dst, chunk, size, 0);
        }
        if (verbose >= 1) {
            std::cout << "\rConversion of uint8 to float is complete!" << std::endl;
        }

        free(buffer_dst);
        free(buffer_src);
        fclose(file_dst);
        fclose(file_src);
    }

    /**
     * Illuminates the pixels in `output` where `mask` is greater than zero.
     * The arrays are assumed to have the same size.
     *
     * @param mask The mask to determine the illumination.
     * @param output The output to illuminate.
     * @param local_flat_size The size of the arrays.
     */
    void illuminate(const uint8_t *__restrict__ mask, float *__restrict__ output, const int64_t local_flat_size) {
        #pragma omp parallel for
        for (int64_t i = 0; i < local_flat_size; i++) {
            if (mask[i] > 0) {
                output[i] = 1.0f;
            }
        }
    }

    /**
     * Perform one step of the diffusion approximation.
     * This means apply the gaussian in each dimension, and then illuminate the pixels where the mask is greater than zero.
     *
     * @param voxels The mask to diffuse.
     * @param buffers The working memory. Should be of size 2.
     * @param N The shape of the mask and the working memory.
     * @param kernel The kernel to apply.
     * @param radius The radius of the kernel.
     */
    void diffusion_step(const uint8_t *__restrict__ voxels, float **buffers, const shape_t &N, const float *__restrict__ kernel, const int64_t radius) {
        for (int64_t dim = 0; dim < 3; dim++) {
            diffusion_core(buffers[0], kernel, buffers[1], N, dim, radius);
            std::swap(buffers[0], buffers[1]);
        }
        illuminate(voxels, buffers[0], N.z*N.y*N.x);
    }

    /**
     * Stores the mask in `mask` where `input` is equal to 1.0f.
     * This is used later to illuminate the pixels.
     * The arrays are assumed to be of the same shape.
     *
     * @param input The input array.
     * @param mask The output mask.
     * @param local_flat_size The size of the arrays.
     */
    void store_mask(const float *__restrict__ input, uint8_t *__restrict__ mask, const int64_t local_flat_size) {
        #pragma omp parallel for
        for (int64_t i = 0; i < local_flat_size; i++) {
            #pragma GCC diagnostic ignored "-Wfloat-equal"
            mask[i] = input[i] == 1.0f ? 1 : 0; // The implant will always be 1.0f
        }
    }

    void diffusion_in_memory(const uint8_t *__restrict__ voxels, const shape_t &N, const float *__restrict__ kernel, const int64_t kernel_size, const int64_t repititions, uint16_t *__restrict__ output, const int verbose) {
        const int64_t
            total_size = N.z*N.y*N.x,
            radius = kernel_size / 2;
        float **buffers = new float*[2];
        buffers[0] = new float[total_size];
        buffers[1] = new float[total_size];

        if (verbose >= 1) {
            std::cout << "Converting uint8 to float" << std::endl;
        }

        convert_uint8_to_float(voxels, buffers[0], total_size);

        for (int64_t rep = 0; rep < repititions; rep++) {
            if (verbose >= 1) {
                std::cout << "\rRepitition: " << rep << "/" << repititions << std::flush;
            }

            diffusion_step(voxels, buffers, N, kernel, radius);
        }

        if (verbose >= 1) {
            std::cout << "\rDiffusion is complete!" << std::endl;
        }

        convert_float_to_uint16(buffers[0], output, total_size);

        if (verbose >= 1) {
            std::cout << "Conversion to uint16 is complete!" << std::endl;
        }

        delete[] buffers[0];
        delete[] buffers[1];
        delete[] buffers;
    }

    void diffusion_on_disk(const std::string &input_file, const float *__restrict__ kernel, const int64_t kernel_size, const std::string &output_file, const shape_t &total_shape, const shape_t &global_shape, const int64_t repititions, const int verbose) {
        std::string
            temp_folder = "/tmp/maxibone",
            temp0 = temp_folder + "/diffusion-temp0.float32",
            temp1 = temp_folder + "/diffusion-temp1.float32";

        if (verbose >= 2) {
            std::cout << "Storing temporary files in: " << temp0 << " and " << temp1 << std::endl;
        }

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

        if (verbose >= 2) {
            std::cout << "Radius: " << radius << std::endl;
            std::cout << "Padding: " << padding << std::endl;
            std::cout << "Disk block size: " << disk_block_size << std::endl;
            std::cout << "Total flat size: " << total_flat_size << std::endl;
            std::cout << "Global blocks: " << global_blocks << std::endl;
            std::cout << "Global flat size: " << global_flat_size << std::endl;
            std::cout << "Global flat size padded: " << global_flat_size_padded << std::endl;
            std::cout << "Layer flat size: " << layer_flat_size << std::endl;
            std::cout << "Disk total flat size: " << disk_total_flat_size << std::endl;
            std::cout << "Disk global flat size: " << disk_global_flat_size << std::endl;
            std::cout << "Disk mask flat size: " << disk_mask_flat_size << std::endl;
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
        convert_uint8_to_float(input_file, temp0, total_flat_size, verbose);

        for (int64_t rep = 0; rep < repititions; rep++) {
            for (int64_t global_block = 0; global_block < global_blocks; global_block++) {
                if (verbose >= 1) {
                    std::cout << "\rRepitition: " << rep << "/" << repititions << " - Global block: " << global_block << "/" << global_blocks << std::flush;
                }

                int64_t this_block_size = std::min(global_shape.z, total_shape.z - global_block*global_shape.z) * layer_flat_size;

                // Load the global block
                load_partial(tmp0, buffers[0], disk_global_flat_size/sizeof(float), global_block*global_shape.z*layer_flat_size, this_block_size, total_flat_size, radius*layer_flat_size);

                // Diffuse the global block
                store_mask(buffers[0], mask, global_flat_size_padded);
                diffusion_step(mask, buffers, global_shape_padded, kernel, radius);

                // Store the global block
                store_partial(tmp1, buffers[0], global_block*global_flat_size, this_block_size, radius*layer_flat_size);
            }
            std::swap(temp0, temp1);
            std::swap(tmp0, tmp1);
        }
        if (verbose >= 1) {
            std::cout << "\rDiffusion is complete!" << std::endl;
        }

        // Convert to uint8
        convert_float_to_uint8(temp0, output_file, total_flat_size, verbose);

        // Free memory
        free(buffers[0]);
        free(buffers[1]);
        delete[] buffers;
        free(mask);
        fclose(tmp0);
        fclose(tmp1);
    }

    #pragma GCC diagnostic ignored "-Wunused-parameter"
    void diffusion_out_of_core(uint8_t *__restrict__ voxels, const shape_t &total_shape, const shape_t &global_shape, const float *__restrict__ kernel, const int64_t kernel_size, const int64_t repititions, uint16_t *__restrict__ output, const int verbose) {
        std::cout << "Not implemented yet" << std::endl;
    }

}