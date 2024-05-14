#ifndef morphology_h
#define morphology_h

#include "datatypes.hh"

namespace NS {

template <typename Op, bool neutral>
void morphology_3d_sphere(
        const mask_type *voxels,
        const int64_t radius,
        const int64_t N[3],
        const int64_t strides[3],
        mask_type *result);


} // namespace NS
#endif