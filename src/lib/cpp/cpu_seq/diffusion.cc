#include <iostream>

#include "diffusion.hh"

namespace cpu_seq {

    #pragma GCC diagnostic ignored "-Wunused-parameter"
    void diffusion_in_memory(const uint8_t *__restrict__ voxels, const shape_t &N, const float *__restrict__ kernel, const int64_t kernel_size, const int64_t repititions, uint8_t *__restrict__ output) {
        std::cout << "Not implemented" << std::endl;
    }

    #pragma GCC diagnostic ignored "-Wunused-parameter"
    void diffusion_on_disk(const std::string &input_file, const float *__restrict__ kernel, const int64_t kernel_size, const std::string &output_file, const shape_t &total_shape, const shape_t &global_shape, const int64_t repititions, const bool verbose) {
        std::cout << "Not implemented yet" << std::endl;
    }

}