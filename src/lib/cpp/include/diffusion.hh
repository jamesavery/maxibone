#ifndef diffusion_h
#define diffusion_h

#include "datatypes.hh"

namespace NS {

    //void convert_float_to_uint8(const std::string &src, const std::string &dst, const int64_t total_flat_size);
    //void convert_uint8_to_float(const std::string &src, const std::string &dst, const int64_t total_flat_size);
    //void diffusion(const std::string &input_file, const std::vector<float>& kernel, const std::string &output_file, const shape_t &total_shape, const shape_t &global_shape, const int64_t repititions, const bool verbose = false);
    //void diffusion_core(const float *__restrict__ input, const float *__restrict__ kernel, float *__restrict__ output, const shape_t &N, const int64_t dim, const int64_t radius, const int64_t padding);
    void diffusion_in_memory(const uint8_t *__restrict__ voxels, const shape_t &N, const double *__restrict__ kernel, const int64_t kernel_size, const int64_t repititions, uint16_t *__restrict__ output);
    //void illuminate(const bool *__restrict__ mask, float *__restrict__ output, const int64_t local_flat_size);
    //void store_mask(const float *__restrict__ input, bool *__restrict__ mask, const int64_t local_flat_size);
    //void stage_to_device(float *__restrict__ stage, const float *__restrict__ src, const idx3drange &range, const shape_t &global_shape, const int64_t kernel_size);
    //void stage_to_host(float *__restrict__ dst, const float *__restrict__ stage, const idx3drange &range, const shape_t &global_shape, const int64_t kernel_size);

}

#endif