#include "morphology.hh"
#include "datatypes.hh"

#include "../cpu_seq/morphology.cc"

namespace cpu_par {

template <typename Op, bool neutral>
void morphology_3d_sphere(
        const mask_type *voxels,
        const int64_t radius,
        const int64_t N[3],
        const int64_t strides[3],
        mask_type *result) {
    cpu_seq::morphology_3d_sphere<Op, neutral>(voxels, radius, N, strides, result);
}

template <uint32_t op(uint32_t,uint32_t), uint32_t reduce(uint32_t,uint32_t), uint32_t neutral>
void morphology_3d_sphere_bitpacked(
        const uint32_t *voxels,
        const int64_t radius,
        const int64_t N[3],
        const int64_t strides[3],
        uint32_t *result) {
    cpu_seq::morphology_3d_sphere_bitpacked<op, reduce, neutral>(voxels, radius, N, strides, result);
}

} // namespace cpu_par