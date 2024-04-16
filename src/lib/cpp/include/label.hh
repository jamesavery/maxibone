#ifndef label_h
#define label_h

#include "datatypes.hh"
#include <chrono>

//typedef uint8_t result_type;
typedef uint16_t result_type;
typedef float_t  prob_type;

namespace NS {

    void material_prob_justonefieldthx(const py::array_t<voxel_type> &np_voxels,
                   const py::array_t<field_type> &np_field,
                   const py::array_t<prob_type>  &np_prob,
                   py::array_t<result_type> &np_result,
                   const std::pair<voxel_type, voxel_type> &vrange,
                   const std::pair<field_type, field_type> &frange,
                   const std::tuple<uint64_t, uint64_t, uint64_t> &offset,
                   const std::tuple<uint64_t, uint64_t, uint64_t> &ranges);

    void otsu(
        const np_array<uint64_t> np_bins,
        np_array<uint64_t> np_result,
        uint64_t step_size);

}

#endif // label_h