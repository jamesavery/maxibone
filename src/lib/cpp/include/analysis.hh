/**
 * @file analysis.hh
 * @brief Various bone analysis functions.
 */
#ifndef analysis_h
#define analysis_h

#include "boilerplate.hh"
#include "datatypes.hh"

namespace NS {

    /**
     * Computes the Bone Implant Contact (BIC) metric for each layer along the first (z) axis. The BIC metric is defined as the ratio of the number of voxels that are both close to the implant surface and classified as soft tissue to the total number of voxels that are close to the implant surface. The distance from the implant surface is determined by the `field` parameter and the `threshold` parameter determines the distance from the implant surface to consider being "close". The `mask` parameter is used to mask out regions of the volume that should not be considered.
     *
     * This function assumes that the output array is already allocated and has the correct shape.
     *
     * @param voxels The segmented voxels, where 1 represents soft tissue and 0 represents everything else (including background). It has shape `(Nz, Ny, Nx)`.
     * @param field The field (either gaussian, EDT or combined). It has shape `(fNz, fNy, fNx)`, where `fNz = Nz / field_scale`, `fNy = Ny / field_scale` and `fNx = Nx / field_scale` and `field_scale >= scale >= 1`.
     * @param mask A binary mask indicating which part of the volume to consider. It has shape `(mNz, mNy, mNx)`, where `mNz = Nz / mask_scale`, `mNy = Ny / mask_scale` and `mNx = Nx / mask_scale` and `mask_scale >= field_scale >= scale >= 1`.
     * @param threshold The threshold for the field. This is the parameter that chooses the distance from the implant surface to consider.
     * @param output The output array. It has shape `(Nz,)`.
     *
     * @return `void`. The result is stored in the `output` array, which is given as a parameter.
     */
    void bic(const input_ndarray<bool> &voxels, const input_ndarray<uint16_t> &field, const input_ndarray<bool> &mask, uint16_t threshold, output_ndarray<float> &output);

} // namespace NS

#endif // analysis_h