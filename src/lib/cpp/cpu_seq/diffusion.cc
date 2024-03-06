#include <iostream>

#include "diffusion.hh"

namespace cpu_seq {

    void diffusion_in_memory(const uint8_t *__restrict__ voxels, const shape_t &N, const float *__restrict__ kernel, const int64_t kernel_size, const int64_t repititions, uint8_t *__restrict__ output) {
        std::cout << "Not implemented" << std::endl;
    }

}