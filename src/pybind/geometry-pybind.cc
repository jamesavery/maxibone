/**
 * @file geometry-pybind.cc
 * @author Carl-Johannes Johnsen (carl-johannes@di.ku.dk)
 * @brief Python bindings for the geometry module.
 * @version 0.1
 * @date 2024-09-17
 *
 * @copyright Copyright (c) 2024
 */
#include "geometry.cc"

namespace python_api {

    /**
     * Computes the center of mass of a mask.
     *
     * @param np_voxels The mask to compute the center of mass of.
     * @return std::array<real_t, 3> The center of mass in zyx-coordinates.
     */
    std::array<real_t, 3> center_of_mass(const np_maskarray &np_voxels) {
        auto voxels_info = np_voxels.request();

        return NS::center_of_mass({voxels_info.ptr, voxels_info.shape});
    }

    /**
     * Computes the center of masses of each material/label in the given tomography.
     * The size of the output array is the number of unique labels in the input array.
     *
     * @param np_voxels The given labeled tomography.
     * @param np_cms The output array of center of masses.
     */
    void center_of_masses(const np_array<uint64_t> &np_voxels, np_array<real_t> &np_cms) {
        auto voxels_info = np_voxels.request();
        auto cms_info = np_cms.request();

        output_ndarray<real_t> cms = {cms_info.ptr, cms_info.shape};

        NS::center_of_masses({voxels_info.ptr, voxels_info.shape}, cms);
    }

    /**
     * Computes the inertia matrix of a mask based on the center of mass.
     *
     * @param np_voxels The mask to compute the inertia matrix of.
     * @param cm The center of mass of the mask.
     * @param verbose The verbosity level. Default is 0.
     * @return std::array<real_t, 9> The inertia matrix.
     */
    std::array<real_t, 9> inertia_matrix(const np_maskarray &np_voxels, std::array<real_t, 3> &cm, const int verbose = 0) {
        auto voxels_info = np_voxels.request();

        return NS::inertia_matrix({voxels_info.ptr, voxels_info.shape}, cm, verbose);
    }

    /**
     * Computes the inertia matrices of the given tomography based of the given center of masses.
     *
     * @param np_voxels The tomography of materials.
     * @param np_cms The center of masses of the materials.
     * @param np_inertia_matrices The output array of inertia matrices.
     */
    void inertia_matrices(const np_array<uint64_t> &np_voxels, const np_array<real_t> &np_cms, np_array<real_t> &np_inertia_matrices) {
        auto voxels_info = np_voxels.request();
        auto cms_info = np_cms.request();
        auto inertia_matrices_info = np_inertia_matrices.request();

        output_ndarray<real_t> inertia_matrices = {inertia_matrices_info.ptr, inertia_matrices_info.shape};

        NS::inertia_matrices({voxels_info.ptr, voxels_info.shape},
            {cms_info.ptr, cms_info.shape},
            inertia_matrices);
    }

    /**
     * Checks which voxels in a 3D array are outside specified ellipsoids. For each voxel, it calculates its distance from the center of mass and determines if it lies outside the ellipsoid defined by the parameters. If a voxel is outside, it increments the corresponding error count in the output array.
     *
     * @param np_voxels The 3D array of voxels.
     * @param np_cms The center of masses of the ellipsoids.
     * @param np_abc The parameters of the ellipsoids.
     * @param errors The output array of error counts.
     */
    void outside_ellipsoid(const np_array<uint64_t> &np_voxels, const np_array<real_t> &np_cms, const np_array<real_t> &np_abc, np_array<uint64_t> errors) {
        auto voxels_info = np_voxels.request();
        auto cms_info = np_cms.request();
        auto abc_info = np_abc.request();
        auto errors_info = errors.request();

        output_ndarray<uint64_t> errors_out = {errors_info.ptr, errors_info.shape};

        NS::outside_ellipsoid({voxels_info.ptr, voxels_info.shape},
            {cms_info.ptr, cms_info.shape},
            {abc_info.ptr, abc_info.shape},
            errors_out);
    }

    /**
     * Samples a plane from a 3D voxel array along specified axes. The function calculates the positions on the plane in voxel index space and samples the voxel values using trilinear interpolation. The sampled values are stored in the `plane_samples`.
     *
     * @param np_voxels The 3D voxel array to sample from.
     * @param voxel_size The size of the voxels in micrometers.
     * @param cm The center of mass of the voxel array.
     * @param u_axis The first axis of the plane.
     * @param v_axis The second axis of the plane.
     * @param bbox The bounding box of the plane in micrometers.
     * @param np_plane_samples The output array of sampled values.
     * @param verbose The verbosity level. Default is 0.
     * @tparam T The type of the voxel values.
     */
    template <typename T>
    void sample_plane(const np_array<T> &np_voxels,
            const real_t voxel_size,
            const std::array<real_t, 3> cm,
            const std::array<real_t, 3> u_axis,
            const std::array<real_t, 3> v_axis,
            const std::array<real_t,4> bbox,
            np_array<real_t> &np_plane_samples,
            const int verbose = 0) {
        auto voxels_info = np_voxels.request();
        auto plane_samples_info  = np_plane_samples.request();

        NS::sample_plane<T>({voxels_info.ptr, voxels_info.shape}, voxel_size,
            cm,u_axis,v_axis,bbox,
            {plane_samples_info.ptr, plane_samples_info.shape},
            verbose);
    }

    /**
     * Integrates values along specified axes of a 3D tomography mask.
     * For each non-zero voxel in the mask, it calculates its position relative to the initial position, projects this position onto the specified axes, and increments the corresponding cell in the output array if the projections are within bounds.
     *
     * @param np_voxels The 3D mask to integrate.
     * @param x0 The initial position in zyx-coordinates.
     * @param v_axis The first axis to integrate along.
     * @param w_axis The second axis to integrate along.
     * @param v_min The minimum value of the first axis.
     * @param w_min The minimum value of the second axis.
     * @param output The output array of integrated values.
     */
    void integrate_axes(const np_maskarray &np_voxels,
            const std::array<real_t, 3> &x0,
            const std::array<real_t, 3> &v_axis,
            const std::array<real_t, 3> &w_axis,
            const real_t v_min, const real_t w_min,
            np_array<uint64_t> &output) {
        auto voxels_info = np_voxels.request();
        auto output_info = output.request();

        NS::integrate_axes({voxels_info.ptr, voxels_info.shape},
            x0,v_axis,w_axis,
            v_min, w_min,
            {output_info.ptr, output_info.shape});
    }

    /**
     * Sets voxels to zero that are outside a bounding box. For each voxel, it calculates its position relative to the center of mass and projects this position onto the principal axes. It then checks if these projections fall outside the specified parameter ranges. If any projection is outside the range, the corresponding voxel value is set to zero.
     *
     * Note that the indices are in voxel space, not micrometers.
     *
     * @param principal_axes The principal axes of the mask.
     * @param parameter_ranges The parameter ranges of the mask.
     * @param cm The center of mass of the mask.
     * @param np_voxels The mask to zero outside the bounding box.
     */
    void zero_outside_bbox(const std::array<real_t, 9> &principal_axes,
            const std::array<real_t, 6> &parameter_ranges,
            const std::array<real_t, 3> &cm,
            np_maskarray &np_voxels) {
        auto voxels_info = np_voxels.request();

        NS::zero_outside_bbox(principal_axes,
            parameter_ranges,
            cm,
            {voxels_info.ptr, voxels_info.shape});
    }

    /**
     * First step of the implant mask filling.
     *
     * The function processes a 3D mask array to compute geometric properties. It initializes pointers to the data of the output arrays for theta values and maximum squared radii. If the `offset` parameter is zero, it sets initial theta values to `pi` and `-pi`. The function then iterates over the mask array, applying a transformation matrix to each voxel's coordinates. For non-zero mask values, it calculates the squared radius and angle in the transformed space, updating the maximum squared radius and theta values accordingly. The function ensures these calculations are within a specified bounding box.
     *
     * @param mask The mask to fill.
     * @param offset The global offset where the mask is located.
     * @param voxel_size The size of the voxels in micrometers.
     * @param bbox The bounding box of the mask in micrometers.
     * @param Muvw The transformation matrix from zyx-coordinates to U'V'W'-coordinates.
     * @param thetas The output array of theta values.
     * @param rsqr_maxs The output array of maximum squared radii.
     */
    void fill_implant_mask_pre(const np_array<mask_type> mask,
            int64_t offset,
            float voxel_size,
            const std::array<float, 6> &bbox,
            const matrix4x4 &Muvw,
            np_array<real_t> thetas,
            np_array<float> rsqr_maxs) {
        auto
            mask_info   = mask.request(),
            thetas_info = thetas.request(),
            rsqr_info   = rsqr_maxs.request();

        return NS::fill_implant_mask_pre({mask_info.ptr, mask_info.shape},
            offset, voxel_size, bbox, Muvw,
            {thetas_info.ptr, thetas_info.shape},
            {rsqr_info.ptr, rsqr_info.shape});
    }

    /**
     * The second step of the implant mask filling.
     *
     * @param implant_mask The mask to fill.
     * @param offset The global offset where the mask is located.
     * @param voxel_size The size of the voxels in micrometers.
     * @param bbox The bounding box of the mask in micrometers.
     * @param r_fraction The fraction of the radius to fill.
     * @param Muvw The transformation matrix from zyx-coordinates to U'V'W'-coordinates.
     * @param thetas The theta values computed in the first step.
     * @param rsqr_maxs The maximum squared radii computed in the first step.
     * @param solid_implant_mask The output mask of the solid implant.
     * @param profile The output array of the profile.
     */
    void fill_implant_mask(const np_maskarray implant_mask,
            int64_t offset,
            float voxel_size,
            const std::array<float, 6> &bbox,
            float r_fraction,
            const matrix4x4 &Muvw,
            const np_array<real_t> thetas,
            const np_array<float> rsqr_maxs,
            np_maskarray solid_implant_mask,
            np_array<float> profile) {
        auto implant_info      = implant_mask.request(),
            thetas_info        = thetas.request(),
            rsqr_info          = rsqr_maxs.request(),
            solid_implant_info = solid_implant_mask.request(),
            profile_info       =  profile.request();

        return NS::fill_implant_mask({implant_info.ptr, implant_info.shape},
            offset, voxel_size, bbox, r_fraction, Muvw,
            {thetas_info.ptr, thetas_info.shape},
            {rsqr_info.ptr, rsqr_info.shape},
            {solid_implant_info.ptr, solid_implant_info.shape},
            {profile_info.ptr, profile_info.shape});
    }

    /**
     * Computes the front mask of the given solid implant mask.
     * The front mask is the mask of the voxels that are in the front of the implant, i.e., the voxels that should contain bone and soft tissue.
     *
     * @param np_solid_implant The solid implant mask.
     * @param voxel_size The size of the voxels in micrometers.
     * @param Muvw The transformation matrix from zyx-coordinates to U'V'W'-coordinates.
     * @param bbox The bounding box of the mask in micrometers.
     * @param np_front_mask The output array of the front mask.
     */
    void compute_front_mask(const np_array<uint8_t> &np_solid_implant,
            const float voxel_size,
            const matrix4x4 &Muvw,
            std::array<float, 6> bbox,
            np_array<mask_type> &np_front_mask) {
        auto solid_implant_info = np_solid_implant.request();
        auto front_mask_info    = np_front_mask.request();

        return NS::compute_front_mask({solid_implant_info.ptr, solid_implant_info.shape},
            voxel_size, Muvw, bbox,
            {front_mask_info.ptr, front_mask_info.shape});
    }

    /**
     * Computes the front, back, implant shell and solid implant masks of the given mask.
     *
     * @param mask The input mask to compute the masks of.
     * @param voxel_size The size of the voxels in micrometers.
     * @param E The Eigen vectors of the mask.
     * @param cm The center of mass of the mask.
     * @param cp The principal axes of the mask.
     * @param UVWp The transformation matrix from zyx-coordinates to U'V'W'-coordinates.
     * @param front_mask The output array of the front mask.
     * @param back_mask The output array of the back mask.
     * @param implant_shell_mask The output array of the implant shell mask.
     * @param solid_implant The output array of the solid implant mask.
     */
    void compute_front_back_masks(const np_array<mask_type> &mask, const float voxel_size, const np_array<float> &E, const np_array<float> &cm, const np_array<float> &cp, const np_array<float> &UVWp, np_array<mask_type> &front_mask, np_array<mask_type> &back_mask, np_array<mask_type> &implant_shell_mask, np_array<mask_type> &solid_implant) {
        auto
            mask_info = mask.request(),
            E_info = E.request(),
            cm_info = cm.request(),
            cp_info = cp.request(),
            UVWp_info = UVWp.request(),
            front_mask_info = front_mask.request(),
            back_mask_info = back_mask.request(),
            implant_shell_mask_info = implant_shell_mask.request(),
            solid_implant_info = solid_implant.request();

        shape_t shape = { mask_info.shape[0], mask_info.shape[1], mask_info.shape[2] };

        const mask_type *mask_ptr = static_cast<const mask_type*>(mask_info.ptr);
        const float
            *E_ptr = static_cast<const float*>(E_info.ptr),
            *cm_ptr = static_cast<const float*>(cm_info.ptr),
            *cp_ptr = static_cast<const float*>(cp_info.ptr),
            *UVWp_ptr = static_cast<const float*>(UVWp_info.ptr);
        mask_type
            *front_mask_ptr = static_cast<mask_type*>(front_mask_info.ptr),
            *back_mask_ptr = static_cast<mask_type*>(back_mask_info.ptr),
            *implant_shell_mask_ptr = static_cast<mask_type*>(implant_shell_mask_info.ptr),
            *solid_implant_ptr = static_cast<mask_type*>(solid_implant_info.ptr);

        return NS::compute_front_back_masks(mask_ptr, shape, voxel_size, E_ptr, cm_ptr, cp_ptr, UVWp_ptr, front_mask_ptr, back_mask_ptr, implant_shell_mask_ptr, solid_implant_ptr);
    }

    /**
     * Projects the given mask to the cylinder FoR.
     *
     * @param np_edt The Euclidean Distance Transform in um, should be low-resolution (will be interpolated)
     * @param np_Cs The Material classification images (probability per voxel, 0..1 -> 0..255)
     * @param Cs_voxel_size The size of voxels in micrometers.
     * @param d_min The minimum distance shell to map to the cylinder.
     * @param d_max The maximum distance shell to map to the cylinder.
     * @param theta_min The minimum angle range (wrt cylinder center).
     * @param theta_max The maximum angle range (wrt cylinder center).
     * @param bbox The implant bounding box (in U'V'W'-coordinates).
     * @param Muvw The transform from zyx (in um) to U'V'W' cylinder FoR (in um).
     * @param np_images The output probability-weighted volume of (class,theta,U)-voxels.
     * @param np_counts N
     * @param verbose The verbosity level. Default is 0.
     */
    void cylinder_projection(const np_array<float> &np_edt,
            const np_bytearray &np_Cs,
            float Cs_voxel_size,
            float d_min, float d_max,
            float theta_min, float theta_max,
            const std::array<float, 6> &bbox,
            const matrix4x4 &Muvw,
            np_array<float> &np_images,
            np_array<uint64_t> &np_counts,
            const int verbose = 0) {
        auto edt_info    = np_edt.request();
        auto Cs_info     = np_Cs.request();
        auto images_info = np_images.request();
        auto counts_info = np_counts.request();

        NS::cylinder_projection({edt_info.ptr,edt_info.shape},
            {Cs_info.ptr, Cs_info.shape},
            Cs_voxel_size,d_min,d_max,theta_min,theta_max,bbox,Muvw,
            {images_info.ptr, images_info.shape},
            {counts_info.ptr, counts_info.shape},
            verbose);
    }

}

PYBIND11_MODULE(geometry, m) {
    m.doc() = "Voxel Geometry Module"; // optional module docstring

    m.def("center_of_mass",           &python_api::center_of_mass, py::arg("np_voxels"));
    m.def("center_of_masses",         &python_api::center_of_masses, py::arg("np_voxels"), py::arg("np_cms").noconvert());
    m.def("inertia_matrix",           &python_api::inertia_matrix, py::arg("np_voxels"), py::arg("cm"), py::arg("verbose") = 0);
    m.def("inertia_matrices",         &python_api::inertia_matrices, py::arg("np_voxels"), py::arg("np_cms"), py::arg("np_inertia_matrices").noconvert());
    m.def("integrate_axes",           &python_api::integrate_axes, py::arg("np_voxels"), py::arg("x0"), py::arg("v_axis"), py::arg("w_axis"), py::arg("v_min"), py::arg("w_min"), py::arg("output").noconvert());
    m.def("outside_ellipsoid",        &python_api::outside_ellipsoid, py::arg("np_voxels"), py::arg("np_cms"), py::arg("np_abc"), py::arg("errors").noconvert());
    m.def("zero_outside_bbox",        &python_api::zero_outside_bbox, py::arg("principal_axes"), py::arg("parameter_ranges"), py::arg("cm"), py::arg("np_voxels").noconvert());
    m.def("fill_implant_mask",        &python_api::fill_implant_mask, py::arg("np_implant_mask"), py::arg("offset"), py::arg("voxel_size"), py::arg("bbox"), py::arg("r_fraction"), py::arg("Muvw"), py::arg("np_thetas").noconvert(), py::arg("np_rsqr_maxs").noconvert(), py::arg("np_solid_implant_mask").noconvert(), py::arg("np_profile").noconvert());
    m.def("fill_implant_mask_pre",    &python_api::fill_implant_mask_pre, py::arg("np_mask"), py::arg("offset"), py::arg("voxel_size"), py::arg("bbox"), py::arg("Muvw"), py::arg("np_thetas").noconvert(), py::arg("np_rsqr_maxs").noconvert());
    m.def("cylinder_projection",      &python_api::cylinder_projection, py::arg("np_edt"), py::arg("np_Cs"), py::arg("Cs_voxel_size"), py::arg("d_min"), py::arg("d_max"), py::arg("theta_min"), py::arg("theta_max"), py::arg("bbox"), py::arg("Muvw"), py::arg("np_images").noconvert(), py::arg("np_counts").noconvert(), py::arg("verbose") = 0);
    m.def("sample_plane",             &python_api::sample_plane<uint16_t>, py::arg("np_voxels"), py::arg("voxel_size"), py::arg("cm"), py::arg("u_axis"), py::arg("v_axis"), py::arg("bbox"), py::arg("np_plano_samples").noconvert(), py::arg("verbose") = 0);
    m.def("sample_plane",             &python_api::sample_plane<uint8_t>, py::arg("np_voxels"), py::arg("voxel_size"), py::arg("cm"), py::arg("u_axis"), py::arg("v_axis"), py::arg("bbox"), py::arg("np_plano_samples").noconvert(), py::arg("verbose") = 0);
    m.def("compute_front_mask",       &python_api::compute_front_mask, py::arg("np_solid_implant"), py::arg("voxel_size"), py::arg("Muvw"), py::arg("bbox"), py::arg("np_front_mask").noconvert());
    m.def("compute_front_back_masks", &python_api::compute_front_back_masks, py::arg("mask"), py::arg("voxel_size"), py::arg("E"), py::arg("cm"), py::arg("cp"), py::arg("UVWp"), py::arg("front_mask").noconvert(), py::arg("back_mask").noconvert(), py::arg("implant_shell_mask").noconvert(), py::arg("solid_implant").noconvert());
}
