/**
 * @file histograms.hh
 * @author Carl-Johannes Johnsen (carl-johannes@di.ku.dk)
 * @brief 2D histogram functions.
 * @version 0.1
 * @date 2024-09-17
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef histograms_h
#define histograms_h

#include "datatypes.hh"

namespace NS {

    /**
     * Computes four 2D histograms of the voxels.
     * A 2D histogram is where each row is a histogram of the voxels in a slice of the 3D volume.
     * E.g. for the x-axis, a histogram is computed for each slice in the x-direction.
     *
     * @param voxels The 3D volume of voxels.
     * @param global_shape The shape of the 3D volume.
     * @param offset The global offset of the block. Used for out-of-core processing.
     * @param block_size The size of the block. Used for out-of-core processing.
     * @param x_bins The output array for the x-axis histograms.
     * @param y_bins The output array for the y-axis histograms.
     * @param z_bins The output array for the z-axis histograms.
     * @param r_bins The output array for the radial histograms.
     * @param voxel_bins The number of bins for each 1D histogram.
     * @param Nr The number of radial bins.
     * @param center The YX center of the volume. Used for computing the radial position of each voxel for the radial histogram.
     * @param vrange The value range of the voxels.
     * @param verbose Whether to print debug information.
     */
    void axis_histogram(const voxel_type __restrict__* voxels,
                        const shape_t &global_shape,
                        const shape_t &offset,
                        const shape_t &block_size,
                        uint64_t __restrict__* x_bins,
                        uint64_t __restrict__* y_bins,
                        uint64_t __restrict__* z_bins,
                        uint64_t __restrict__* r_bins,
                        const uint64_t voxel_bins,
                        const uint64_t Nr,
                        const std::tuple<uint64_t, uint64_t> &center,
                        const std::tuple<double, double> &vrange,
                        const bool verbose);

    /**
     * Computes a 2D field histogram.
     * A field histogram is where for each `v = voxel[z,y,x]` value and corresponding `f = field[z,y,x]` value, index `i,j` in the resulting histogram is incremented, where `i` is the field bin, and `j` is the voxel bin.
     * If the field shape is less than the voxel shape, the field is upsampled to the voxel shape.
     *
     * @param voxels The 3D volume of voxels.
     * @param field The 3D volume of fields.
     * @param voxels_shape The shape of the 3D volume of voxels.
     * @param field_shape The shape of the 3D volume of fields.
     * @param offset The global offset of the block. Used for out-of-core processing.
     * @param block_size The size of the block. Used for out-of-core processing.
     * @param bins The output array for the histogram.
     * @param voxel_bins The number of bins for the voxel histogram.
     * @param field_bins The number of bins for the field histogram.
     * @param vrange The value range of the voxels.
     * @param frange The value range of the fields.
     * @param verbose Whether to print debug information.
     */
    void field_histogram(const voxel_type *__restrict__ voxels,
                         const field_type *__restrict__ field,
                         const shape_t &voxels_shape,
                         const shape_t &field_shape,
                         const shape_t &offset,
                         const shape_t &block_size,
                         uint64_t *__restrict__ bins,
                         const uint64_t voxel_bins,
                         const uint64_t field_bins,
                         const std::tuple<double, double> &vrange,
                         const std::tuple<double, double> &frange,
                         const bool verbose);

    /**
     * Computes a 2D field histogram.
     * A field histogram is where for each `v = voxel[z,y,x]` value and corresponding `f = field[z,y,x]` value, index `i,j` in the resulting histogram is incremented, where `i` is the field bin, and `j` is the voxel bin.
     * This function differs from `field_histogram` in that it resamples the field to the voxel shape using the neighbouring field values, rather than just choosing the closest.
     *
     * @param voxels The 3D volume of voxels.
     * @param field The 3D volume of fields.
     * @param voxels_shape The shape of the 3D volume of voxels.
     * @param field_shape The shape of the 3D volume of fields.
     * @param offset The global offset of the block. Used for out-of-core processing.
     * @param block_size The size of the block. Used for out-of-core processing.
     * @param bins The output array for the histogram.
     * @param voxel_bins The number of bins for the voxel histogram.
     * @param field_bins The number of bins for the field histogram.
     * @param vrange The value range of the voxels.
     * @param frange The value range of the fields.
     * @param verbose Whether to print debug information.
     */
    void field_histogram_resample(const voxel_type *__restrict__ voxels,
                                  const field_type *__restrict__ field,
                                  const shape_t &voxels_shape,
                                  const shape_t &field_shape,
                                  const shape_t &offset,
                                  const shape_t &block_size,
                                  uint64_t *__restrict__ &bins,
                                  const uint64_t voxel_bins,
                                  const uint64_t field_bins,
                                  const std::tuple<double, double> &vrange,
                                  const std::tuple<double, double> &frange,
                                  const bool verbose);

}

/**
 * Checks whether the point `U,V,W` is inside the bounding box `bbox`.
 *
 * @param U The U coordinate.
 * @param V The V coordinate.
 * @param W The W coordinate.
 * @param bbox The bounding box.
 * @return `true` if the point is inside the bounding box, `false` otherwise.
 */
inline bool in_bbox(float U, float V, float W, const std::array<float, 6> bbox) {
    const auto& [U_min,U_max,V_min,V_max,W_min,W_max] = bbox;

    return
        U >= U_min && U <= U_max &&
        V >= V_min && V <= V_min &&
        W >= W_min && W <= W_max;
}

/**
 * Resamples the voxels at the point `X` using trilinear interpolation.
 *
 * @param voxels The 3D volume of voxels.
 * @param shape The shape of the 3D volume.
 * @param X The point to resample at.
 * @return float The resampled value.
 */
inline float resample2x2x2(const field_type __restrict__* voxels,
                            const shape_t &shape,
                            const std::array<float,3> &X) {
    auto [Nz,Ny,Nx] = shape;

    if (!in_bbox(X[0], X[1], X[2],
            {0.5f, (float)Nx-1.5f, 0.5f, (float)Ny-1.5f, 0.5f, (float)Nz-1.5f})) {
        uint64_t voxel_index = ((uint64_t) X[0])*Ny*Nz + ((uint64_t)floor(X[1]))*Ny+((uint64_t)floor(X[2]));
        return voxels[voxel_index];
    }

    float   Xfrac[2][3];	// {Xminus[3], Xplus[3]}
    int64_t  Xint[2][3];	// {Iminus[3], Iplus[3]}
    float   value = 0;

    for(int i = 0; i < 3; i++) {
        double Iminus, Iplus;
        Xfrac[0][i] = 1-(float) modf(X[i]-0.5, &Iminus); // 1-{X[i]-1/2}, floor(X[i]-1/2)
        Xfrac[1][i] =   (float) modf(X[i]+0.5, &Iplus);  // {X[i]+1/2}, floor(X[i]+1/2)

        Xint[0][i] = (int64_t) Iminus;
        Xint[1][i] = (int64_t) Iplus;
    }

    for(int ijk = 0; ijk <= 7; ijk++) {
        float  weight = 1;
        int64_t IJK[3] = { 0, 0, 0 };

        for(int axis = 0; axis < 3; axis++) { // x-1/2 or x+1/2
            int pm = (ijk >> axis) & 1;
            IJK[axis] = Xint[pm][axis];
            weight   *= Xfrac[pm][axis];
        }

        auto [I,J,K] = IJK;
        if(I<0 || J<0 || K<0) {
            printf("(I,J,K) = (%ld,%ld,%ld)\n", I, J, K);

            abort();
        }

        if (I >= int(Nx) || J >= int(Ny) || K >= int(Nz)) {
            printf("(I,J,K) = (%ld,%ld,%ld), (Nx,Ny,Nz) = (%ld,%ld,%ld)\n", I, J, K, Nx, Ny, Nz);
            abort();
        }

        uint64_t voxel_index = I*Ny*Nz + J*Ny + K;
        field_type voxel = voxels[voxel_index];
        value += voxel * weight;
    }

    return value;
}

#endif