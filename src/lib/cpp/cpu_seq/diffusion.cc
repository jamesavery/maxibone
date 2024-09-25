/**
 * @file diffusion.cc
 * @author Carl-Johannes Johnsen (carl-johannes@di.ku.dk)
 * @brief Sequential CPU implementation of the diffusion approximation algorithm.
 * @version 0.1
 * @date 2024-09-16
 *
 * @copyright Copyright (c) 2024
 */
#include "diffusion.hh"

namespace cpu_seq {

    #pragma GCC diagnostic ignored "-Wunused-parameter"
    void diffusion_in_memory(const uint8_t *__restrict__ voxels, const shape_t &N, const float *__restrict__ kernel, const int64_t kernel_size, const int64_t repititions, uint16_t *__restrict__ output, const int verbose) {
        std::cout << "Not implemented" << std::endl;
    }

    #pragma GCC diagnostic ignored "-Wunused-parameter"
    void diffusion_on_disk(const std::string &input_file, const float *__restrict__ kernel, const int64_t kernel_size, const std::string &output_file, const shape_t &total_shape, const shape_t &global_shape, const int64_t repititions, const int verbose) {
        std::cout << "Not implemented yet" << std::endl;
    }

    #pragma GCC diagnostic ignored "-Wunused-parameter"
    void diffusion_out_of_core(uint8_t *__restrict__ voxels, const shape_t &total_shape, const shape_t &global_shape, const float *__restrict__ kernel, const int64_t kernel_size, const int64_t repititions, uint16_t *__restrict__ output, const int verbose) {
        std::cout << "Not implemented yet" << std::endl;
    }

}