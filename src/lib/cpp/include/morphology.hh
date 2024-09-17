/**
 * @file morphology.hh
 * @author Carl-Johannes Johnsen (carl-johannes@di.ku.dk)
 * @brief Morphology operations on 3D masks.
 * @version 0.1
 * @date 2024-09-17
 *
 * @copyright Copyright (c) 2024
 */
#ifndef morphology_h
#define morphology_h

#include "datatypes.hh"

namespace NS {

    /**
     * 3D sphere morphology operation performed on a 3D mask.
     *
     * @param voxels The input mask.
     * @param radius The radius of the sphere.
     * @param N The shape of the mask.
     * @param strides The strides of the mask.
     * @param result The output mask.
     * @tparam Op The operation to perform. Should be either `std::bit_and` for erosion or `std::bit_or` for dilation.
     * @tparam neutral The neutral element for the operation. Should be either `true` for erosion or `false` for dilation.
     */
    template <typename Op, bool neutral>
    void morphology_3d_sphere(
        const mask_type *voxels,
        const int64_t radius,
        const int64_t N[3],
        const int64_t strides[3],
        mask_type *result);

    /**
     * 3D sphere morphology operation performed on a 3D mask with a radius of 16.
     * This is a special case of the general 3D sphere morphology operation, and the hardcoding of the radius allows for more optimizations by the compiler.
     *
     * @param voxels The input mask.
     * @param N The shape of the mask.
     * @param strides The strides of the mask.
     * @param result The output mask.
     * @tparam Op The operation to perform. Should be either `std::bit_and` for erosion or `std::bit_or` for dilation.
     * @tparam neutral The neutral element for the operation. Should be either `true` for erosion or `false` for dilation.
     */
    template <typename Op, bool neutral>
    void morphology_3d_sphere_r16(
        const mask_type *voxels,
        const int64_t N[3],
        const int64_t strides[3],
        mask_type *result);

    /**
     * Bitpacked 3D sphere morphology operation performed on a 3D mask.
     * The operation is performed on a bitpacked mask, where each bit represents a voxel.
     *
     * See `bitpacking.hh` for more information on bitpacking.
     *
     * @param voxels The input mask.
     * @param radius The radius of the sphere.
     * @param N The shape of the mask.
     * @param strides The strides of the mask.
     * @param result The output mask.
     * @tparam op(uint32_t, uint32_t) The operation to perform.
     * @tparam reduce(uint32_t, uint32_t) The reduction operation to perform.
     * @tparam neutral The neutral element for the operation.
     */
    template <uint32_t op(uint32_t,uint32_t), uint32_t reduce(uint32_t,uint32_t), uint32_t neutral>
    void morphology_3d_sphere_bitpacked(
        const uint32_t *voxels,
        const int64_t radius,
        const int64_t N[3],
        const int64_t strides[3],
        uint32_t *result);

} // namespace NS

#endif // morphology_h