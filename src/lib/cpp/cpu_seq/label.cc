/**
 * @file label.cc
 * @author Carl-Johannes (carl-johannes@di.ku.dk)
 * @brief CPU sequential implementations of the functions used in material segmentation.
 * @version 0.1
 * @date 2024-09-17
 *
 * @copyright Copyright (c) 2024
 */
#include "label.hh"

namespace cpu_seq {

    #pragma GCC diagnostic ignored "-Wunused-parameter"
    void material_prob_justonefieldthx(const py::array_t<voxel_type> &np_voxels,
            const py::array_t<field_type> &np_field,
            const py::array_t<prob_type> &np_prob,
            py::array_t<result_type> &np_result,
            const std::pair<voxel_type, voxel_type> &vrange,
            const std::pair<field_type, field_type> &frange,
            const std::tuple<uint64_t, uint64_t, uint64_t> &offset,
            const std::tuple<uint64_t, uint64_t, uint64_t> &ranges,
            const int verbose) {
        throw std::runtime_error("Not implemented");
    }

    #pragma GCC diagnostic ignored "-Wunused-parameter"
    void otsu(
            const np_array<uint64_t> &np_bins,
            np_array<uint64_t> &np_result,
            uint64_t step_size) {
        throw std::runtime_error("Not implemented");
    }

}