/**
 * @file geometry.hh
 * @author Carl-Johannes Johnsen (carl-johannes@di.ku.dk)
 * @brief Geometry-related functions.
 * @version 0.1
 * @date 2024-09-16
 *
 * @copyright Copyright (c) 2024
 */
#ifndef geometry_h
#define geometry_h

#include "datatypes.hh"

/**
 * Computes the dot product of two vectors of size 3.
 *
 * @param a The first vector.
 * @param b The second vector.
 */
#define DOT(a,b) (a[0]*b[0] + a[1]*b[1] + a[2]*b[2])

/**
 * Prints a message to stderr with a timestamp.
 *
 * @param message the message to print.
 */
inline void print_timestamp(std::string message) {
    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    tm local_tm = *localtime(&now);
    fprintf(stderr, "%s at %02d:%02d:%02d\n", message.c_str(), local_tm.tm_hour, local_tm.tm_min, local_tm.tm_sec);
}

/**
 * Homogeneous transformation on a 4-dimensional vector using a 4x4 transformation matrix. It multiplies the matrix by the vector to produce a new transformed vector.
 *
 * @param x the vector to transform.
 * @param M the transformation matrix.
 * @return `vector4` the transformed vector.
 */
inline vector4 hom_transform(const vector4 &x, const matrix4x4 &M) {
    vector4 c{{ 0, 0, 0, 0 }};

    for (int i = 0; i < 4; i++) {
        real_t sum = 0;
        for (int j = 0; j < 4; j++) {
            sum += M[i*4 + j] * x[j];
        }
        c[i] = sum;
    }
    return c;
}

/**
 * Checks whether a given index is inside a given bounding box.
 *
 * @param index the index to check.
 * @param bbox the bounding box.
 * @return `true` if the index is inside the bounding box, `false` otherwise.
 */
inline bool in_bbox(const std::array<float, 3> index, const std::array<float, 6> &bbox) {
    const auto& [z, y, x] = index;
    const auto& [zmin, zmax, ymin, ymax, xmin, xmax] = bbox;

    return
        z >= zmin && z <= zmax &&
        y >= ymin && y <= ymax &&
        x >= xmin && x <= xmax;
}

/**
 * Resamples a 3D array at a given index using trilinear interpolation.
 *
 * @param voxels the 3D array to resample.
 * @param shape the shape of the 3D array.
 * @param index the index to resample at.
 * @return `float` the resampled value.
 */
template <typename T>
float resample2x2x2(const T                      *voxels,
                    const std::array<ssize_t, 3> &shape,
                    const std::array<float, 3>   &index) {
    auto  [Nz,Ny,Nx] = shape;

    if (!in_bbox(index, {0.5f, float(Nx)-0.5f, 0.5f, float(Ny)-0.5f, 0.5f, float(Nz)-0.5f})) {
        uint64_t voxel_index = uint64_t(floor(index[0]))*Ny*Nx + uint64_t(floor(index[1]))*Nx + uint64_t(floor(index[2]));
        return voxels[voxel_index];
    }

    float   Ifrac[2][3]; // {Xminus[3], Xplus[3]}
    int64_t Iint[2][3];  // {Iminus[3], Iplus[3]}
    float   value = 0;

    for (int i = 0; i < 3; i++) {
        float Iminus, Iplus;
        Ifrac[0][i] = 1-std::modf(index[i]-0.5f, &Iminus); // 1-{X[i]-1/2}, floor(X[i]-1/2)
        Ifrac[1][i] =   std::modf(index[i]+0.5f, &Iplus);  // {X[i]+1/2}, floor(X[i]+1/2)

        Iint[0][i] = (int64_t) Iminus;
        Iint[1][i] = (int64_t) Iplus;
    }

    for (int ijk = 0; ijk <= 7; ijk++) {
        float  weight = 1;
        int64_t IJK[3] = {0,0,0};

        for (int axis = 0; axis < 3; axis++) { // x-1/2 or x+1/2
            int pm    = (ijk >> axis) & 1;
            IJK[axis] = Iint[pm][axis];
            weight   *= Ifrac[pm][axis];
        }

        auto [I,J,K] = IJK;
        // TODO DEBUG macro
        // if (I<0 || J<0 || K<0) {
        //   printf("(I,J,K) = (%ld,%ld,%ld)\n",I,J,K);
        //   abort();
        // }
        // if (I>=int(Nx) || J>=int(Ny) || K>=int(Nz)) {
        //   printf("(I,J,K) = (%ld,%ld,%ld), (Nx,Ny,Nz) = (%ld,%ld,%ld)\n",I,J,K,Nx,Ny,Nz);
        //   abort();
        // }
        uint64_t voxel_index = I*Ny*Nx + J*Nx + K;
        //assert(I>=0 && J>=0 && K>=0);
        //assert(I<Nx && J<Ny && K<Nz);
        float voxel = (float) voxels[voxel_index];
        value += voxel * weight;
    }

    return value;
}

namespace NS {

    /**
     * Computes the center of mass of the given tomography.
     *
     * @param voxels The given tomography.
     * @return `vector3` The center of mass.
     */
    std::array<real_t,3> center_of_mass(const input_ndarray<mask_type> &voxels);

    /**
     * Computes the center of masses of each material/label in the given tomography.
     * The size of the output array is the number of unique labels in the input array.
     *
     * @param mask The given tomography.
     * @param output The center of masses.
     */
    void center_of_masses(const input_ndarray<uint64_t> &mask, output_ndarray<real_t> &output);

    /**
     * Computes the front mask of the given solid implant mask.
     * The front mask is the mask of the voxels that are in the front of the implant, i.e., the voxels that should contain bone and soft tissue.
     *
     * @param solid_implant the solid implant mask.
     * @param voxel_size the size of the voxels in micrometers.
     * @param Muvw the transformation matrix from the zyx (in um) to the UVW (in um) cylinder FoR.
     * @param bbox the bounding box of the solid implant mask.
     * @param front_mask the output front mask.
     */
    void compute_front_mask(const input_ndarray<mask_type> solid_implant, const float voxel_size, const matrix4x4 &Muvw, std::array<float,6> bbox, output_ndarray<mask_type> front_mask);

    /**
     * Computes the front, back, implant shell and solid implant masks of the given mask.
     *
     * @param mask the given mask.
     * @param shape the shape of the mask.
     * @param voxel_size the size of the voxels in micrometers.
     * @param E the Eigen vectors of the mask.
     * @param cm the center of mass of the mask.
     * @param cp the principal axes of the mask.
     * @param UVWp the transformation matrix from the zyx (in um) to the UVW (in um) cylinder FoR.
     * @param front_mask the output front mask.
     * @param back_mask the output back mask.
     * @param implant_shell_mask the output implant shell mask.
     * @param solid_implant the output solid implant mask.
     */
    void compute_front_back_masks(const mask_type *mask, const shape_t &shape, const float voxel_size, const float *E, const float *cm, const float *cp, const float *UVWp, mask_type *front_mask, mask_type *back_mask, mask_type *implant_shell_mask, mask_type *solid_implant);

    /**
     * Projects the given mask to the cylinder FoR.
     *
     * @param edt Euclidean Distance Transform in um, should be low-resolution (will be interpolated)
     * @param C Material classification images (probability per voxel, 0..1 -> 0..255)
     * @param voxel_size the size of the voxels in micrometers.
     * @param d_min Minimum distance shell to map to cylinder
     * @param d_max Maximum distance shell to map to cylinder
     * @param theta_min Minimum of the angle range (wrt cylinder center)
     * @param theta_max Maximum of the angle range (wrt cylinder center)
     * @param bbox the bounding box of the solid implant mask.
     * @param Muvw The transform from zyx (in um) to U'V'W' cylinder FoR (in um)
     * @param image Probability-weighted volume of (class,theta,U)-voxels
     * @param count Number of (class,theta,U)-voxels
     */
    void cylinder_projection(const input_ndarray<float> edt, const input_ndarray<uint8_t> C, float voxel_size, float d_min, float d_max, float theta_min, float theta_max, std::array<float,6> bbox, const matrix4x4 &Muvw, output_ndarray<float>    image, output_ndarray<int64_t> count);

    /**
     * First step of the implant mask filling.
     *
     * The function processes a 3D mask array to compute geometric properties. It initializes pointers to the data of the output arrays for theta values and maximum squared radii. If the `offset` parameter is zero, it sets initial theta values to `pi` and `-pi`. The function then iterates over the mask array, applying a transformation matrix to each voxel's coordinates. For non-zero mask values, it calculates the squared radius and angle in the transformed space, updating the maximum squared radius and theta values accordingly. The function ensures these calculations are within a specified bounding box.
     *
     * @param mask The mask to fill.
     * @param offset The global offset where the mask resides.
     * @param voxel_size The size of voxels in micrometers.
     * @param bbox The bounding box of the mask.
     * @param Muvw The transformation matrix from the zyx (in um) to the UVW (in um) cylinder FoR.
     * @param thetas The output thetas.
     * @param rsqr_maxes The output rsqr_maxes.
     */
    void fill_implant_mask_pre(const input_ndarray<mask_type> mask, int64_t offset, float voxel_size, const std::array<float,6> &bbox, const matrix4x4 &Muvw, output_ndarray<real_t> thetas, output_ndarray<float> rsqr_maxes);

    /**
     * The second step of the implant mask filling.
     *
     * @param mask The mask to fill.
     * @param offset The global offset where the mask resides.
     * @param voxel_size The size of voxels in micrometers.
     * @param bbox The bounding box of the mask.
     * @param r_fraction The fraction of the radius to use.
     * @param Muvw The transformation matrix from the zyx (in um) to the UVW (in um) cylinder FoR.
     * @param thetas The thetas computed in the first step.
     * @param rsqr_maxs The rsqr_maxes computed in the first step.
     * @param solid_implant_mask The output solid implant mask.
     * @param profile The output profile of the implant.
     */
    void fill_implant_mask(const input_ndarray<mask_type> mask, int64_t offset, float voxel_size, const std::array<float,6> &bbox, float r_fraction, const matrix4x4 &Muvw, const input_ndarray<real_t> thetas, const input_ndarray<float> rsqr_maxs, output_ndarray<mask_type> solid_implant_mask, output_ndarray<float> profile);

    /**
     * Computes the inertia matrix of the given tomography based of the given center of mass.
     *
     * @param voxels The given tomography.
     * @param cm The given center of mass.
     * @returns `array<real_t,9>` The 3x3 inertia matrix.
     */
    std::array<real_t,9> inertia_matrix(const input_ndarray<mask_type> &voxels, const std::array<real_t,3> &cm);

    /**
     * Computes the inertia matrices of the given tomography based of the given center of masses.
     *
     * @param mask The given tomography of materials.
     * @param cms The given center of masses.
     * @param output The output inertia matrices.
     */
    void inertia_matrices(const input_ndarray<uint64_t> &mask, const input_ndarray<real_t> &cms, output_ndarray<real_t> &output);

    /**
     * Integrates values along specified axes of a 3D tomography mask.
     * For each non-zero voxel in the mask, it calculates its position relative to the initial position, projects this position onto the specified axes, and increments the corresponding cell in the output array if the projections are within bounds.
     *
     * @param mask The given tomography.
     * @param x0 The initial position.
     * @param v_axis The first axis to integrate along.
     * @param w_axis The second axis to integrate along.
     * @param v_min The minimum value of the first axis.
     * @param w_min The minimum value of the second axis.
     * @param output The output 2D array of integrated values.
     */
    void integrate_axes(const input_ndarray<mask_type> &mask, const std::array<real_t,3> &x0, const std::array<real_t,3> &v_axis, const std::array<real_t,3> &w_axis, const real_t v_min, const real_t w_min, output_ndarray<uint64_t> output);

    /**
     * Checks which voxels in a 3D array are outside specified ellipsoids. For each voxel, it calculates its distance from the center of mass and determines if it lies outside the ellipsoid defined by the parameters. If a voxel is outside, it increments the corresponding error count in the output array.
     *
     * @param voxels The 3D array of voxels.
     * @param cms The center of mass of the ellipsoids.
     * @param abc The parameters of the ellipsoids.
     * @param errors The output array of error counts.
     */
    void outside_ellipsoid(const input_ndarray<uint64_t> &voxels, const input_ndarray<real_t> &cms, const input_ndarray<real_t> &abc, output_ndarray<uint64_t> &errors);

    /**
     * Samples a plane from a 3D voxel array along specified axes. The function calculates the positions on the plane in voxel index space and samples the voxel values using trilinear interpolation. The sampled values are stored in the `plane_samples`.
     *
     * @param voxels The 3D array of voxels.
     * @param voxel_size The size of the voxels in micrometers.
     * @param cm The center of mass of the voxels.
     * @param u_axis The first axis of the plane.
     * @param v_axis The second axis of the plane.
     * @param bbox The bounding box of the plane in micrometers.
     * @param plane_samples The output array of sampled values.
     * @tparam T The element type of the input voxels.
     */
    template <typename T>
    void sample_plane(const input_ndarray<T> &voxels, const real_t voxel_size, const std::array<real_t, 3> cm, const std::array<real_t, 3> u_axis, const std::array<real_t, 3> v_axis, const std::array<real_t, 4>  bbox, output_ndarray<real_t> plane_samples);

    /**
     * Sets voxels to zero that are outside a bounding box. For each voxel, it calculates its position relative to the center of mass and projects this position onto the principal axes. It then checks if these projections fall outside the specified parameter ranges. If any projection is outside the range, the corresponding voxel value is set to zero.
     *
     * Note that the indices are in voxel space, not micrometers.
     *
     * @param principal_axes The principal axes of the voxels.
     * @param parameter_ranges The parameter ranges of the voxels.
     * @param cm The center of mass of the voxels.
     * @param voxels The 3D array of voxels.
     */
    void zero_outside_bbox(const std::array<real_t,9> &principal_axes, const std::array<real_t,6> &parameter_ranges, const std::array<real_t,3> &cm, output_ndarray<mask_type> voxels);

}

#endif