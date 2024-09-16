/**
 * @file diffusion.hh
 * @author Carl-Johannes Johnsen (carl-johannes@di.ku.dk)
 * @brief 3D Diffusion approximation using repeated applications of a 1D gaussian in each dimension.
 * @version 0.1
 * @date 2024-09-16
 *
 * @copyright Copyright (c) 2024
 */
#ifndef diffusion_h
#define diffusion_h

#include "datatypes.hh"

namespace NS {

    //
    // External functions (i.e. assumed to be called through the Python bindings).
    //

    /**
     * In-memory 3D diffusion approximation using repeated applications of a 1D gaussian in each dimension.
     * After each application in all dimensions, the input "illuminates" the set pixels, so that they do not loose intensity.
     * This function assumes that the target device has enough memory to hold both input, output and working memory.
     * This translates to `N.z*N.y*N.x * (sizeof(uint8_t) + 2*sizeof(float) + sizeof(uint16_t)) + kernel_size`.
     * This function also assumes that the output is preallocated and has the same shape as `voxels`
     *
     * @param voxels Pointer to the mask. Assumed to be either 1 or 0.
     * @param N Shape of the mask.
     * @param kernel Pointer to the 1D gaussian kernel to apply.
     * @param kernel_size The size of the kernel.
     * @param repititions The number of repititions to apply the kernel.
     * @param output Pointer to where the result should be stored.
     */
    void diffusion_in_memory(const uint8_t *__restrict__ voxels, const shape_t &N, const float *__restrict__ kernel, const int64_t kernel_size, const int64_t repititions, uint16_t *__restrict__ output);

    /**
     * Out-of-core 3D diffusion approximation using repeated applications of a 1D gaussian in each dimension.
     * After each application in all dimensions, the input "illuminates" the set pixels, so that they do not loose intensity.
     * This function keeps the intermediate results on disk, as they cannot even fit in main memory.
     * The target device should be able to fit one chunk of size `global_shape`.
     * As such, this function relies heavily on fast disk access.
     * This function also assumes that the output is preallocated and has the same shape as `voxels`
     *
     * @param input_file The path to the input file.
     * @param kernel The 1D gaussian to apply.
     * @param kernel_size The size of the kernel.
     * @param output_file The path to the output file.
     * @param total_shape The shape of the volume.
     * @param global_shape The shape of each chunk to load in and out.
     * @param repititions How many times to apply the gaussians.
     * @param verbose Whether debug information should be printed.
     */
    void diffusion_on_disk(const std::string &input_file, const float *__restrict__ kernel, const int64_t kernel_size, const std::string &output_file, const shape_t &total_shape, const shape_t &global_shape, const int64_t repititions, const bool verbose);

    /**
     * Out-of-core 3D diffusion approximation using repeated applications of a 1D gaussian in each dimension.
     * After each application in all dimensions, the input "illuminates" the set pixels, so that they do not loose intensity.
     * This function assumes that the host has enough memory to hold both input, output and working memory.
     * This translates to `N.z*N.y*N.x * (sizeof(uint8_t) + 2*sizeof(float) + sizeof(uint16_t)) + kernel_size`.
     * The target will only have to be able to hold a chunk of size `global_shape`.
     *
     * @param voxels Pointer to the mask. Assumed to be either 1 or 0.
     * @param total_shape The shape of the volume.
     * @param global_shape The shape of each chunk to load in and out.
     * @param kernel Pointer to the 1D gaussian kernel to apply.
     * @param kernel_size The size of the kernel.
     * @param repititions The number of repititions to apply the kernel.
     * @param output Pointer to where the result should be stored.
     */
    void diffusion_out_of_core(uint8_t *__restrict__ voxels, const shape_t &total_shape, const shape_t &global_shape, const float *__restrict__ kernel, const int64_t kernel_size, const int64_t repititions, uint16_t *__restrict__ output);

    //
    // Internal functions (i.e. assumed to be called internally in C++).
    //

}

#endif