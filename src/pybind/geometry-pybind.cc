#include "geometry.cc"

namespace python_api {

array<real_t, 3> center_of_mass(const np_maskarray &np_voxels) {
    auto voxels_info = np_voxels.request();

    return NS::center_of_mass({voxels_info.ptr, voxels_info.shape});
}

void center_of_masses(const np_array<uint64_t> &np_voxels, np_array<real_t> &np_cms) {
    auto voxels_info = np_voxels.request();
    auto cms_info = np_cms.request();

    output_ndarray<real_t> cms = {cms_info.ptr, cms_info.shape};

    NS::center_of_masses({voxels_info.ptr, voxels_info.shape},
            cms);
}

array<real_t, 9> inertia_matrix(const np_maskarray &np_voxels, array<real_t, 3> &cm) {
    auto voxels_info = np_voxels.request();

    return NS::inertia_matrix({voxels_info.ptr, voxels_info.shape}, cm);
}

void inertia_matrices(const np_array<uint64_t> &np_voxels, const np_array<real_t> &np_cms, np_array<real_t> &np_inertia_matrices) {
    auto voxels_info = np_voxels.request();
    auto cms_info = np_cms.request();
    auto inertia_matrices_info = np_inertia_matrices.request();

    output_ndarray<real_t> inertia_matrices = {inertia_matrices_info.ptr, inertia_matrices_info.shape};

    NS::inertia_matrices({voxels_info.ptr, voxels_info.shape},
            {cms_info.ptr, cms_info.shape},
            inertia_matrices);
}


template <typename T>
void sample_plane(const np_array<T> &np_voxels,
          const real_t voxel_size, // In micrometers
          const array<real_t,3> cm,
          const array<real_t,3> u_axis,
          const array<real_t,3> v_axis,
          const array<real_t,4>  bbox,    // [umin,umax,vmin,vmax] in micrometers
          np_array<real_t> &np_plane_samples) {
    auto voxels_info = np_voxels.request();
    auto plane_samples_info  = np_plane_samples.request();

    NS::sample_plane<T>({voxels_info.ptr, voxels_info.shape}, voxel_size,
           cm,u_axis,v_axis,bbox,
           {plane_samples_info.ptr, plane_samples_info.shape});
}

void integrate_axes(const np_maskarray &np_voxels,
            const array<real_t,3> &x0,
            const array<real_t,3> &v_axis,
            const array<real_t,3> &w_axis,
            const real_t v_min, const real_t w_min,
            np_array<uint64_t> &output) {
    auto voxels_info = np_voxels.request();
    auto output_info = output.request();

    NS::integrate_axes({voxels_info.ptr, voxels_info.shape},
             x0,v_axis,w_axis,
             v_min, w_min,
             {output_info.ptr, output_info.shape});
}

void zero_outside_bbox(const array<real_t,9> &principal_axes,
             const array<real_t,6> &parameter_ranges,
             const array<real_t,3> &cm, // TOOD: Med eller uden voxelsize?
             np_maskarray &np_voxels) {
    auto voxels_info = np_voxels.request();

    NS::zero_outside_bbox(principal_axes,
              parameter_ranges,
              cm,
              {voxels_info.ptr, voxels_info.shape});
}

void fill_implant_mask_pre(const np_array<mask_type> mask,
               int64_t offset,
               float voxel_size,
               const array<float,6> &bbox,
               const matrix4x4 &Muvw,
               np_array<real_t> thetas,
               np_array<float> rsqr_maxs) {
    auto
        mask_info    = mask.request(),
        thetas_info  = thetas.request(),
        rsqr_info    = rsqr_maxs.request();

    return NS::fill_implant_mask_pre({mask_info.ptr, mask_info.shape},
                 offset, voxel_size, bbox, Muvw,
                 {thetas_info.ptr, thetas_info.shape},
                 {rsqr_info.ptr, rsqr_info.shape});
}

void fill_implant_mask(const np_maskarray implant_mask,
               int64_t offset,
               float voxel_size,
               const array<float,6> &bbox,
               float r_fraction,
               const matrix4x4 &Muvw,
               const np_array<real_t> thetas,
               const np_array<float> rsqr_maxs,
               np_maskarray solid_implant_mask,
               np_array<float> profile
               ) {
    auto implant_info      = implant_mask.request(),
        thetas_info        = thetas.request(),
        rsqr_info          = rsqr_maxs.request(),
        solid_implant_info = solid_implant_mask.request(),
        profile_info       =  profile.request();

    return NS::fill_implant_mask({implant_info.ptr,       implant_info.shape},
                 offset, voxel_size, bbox, r_fraction, Muvw,
                 {thetas_info.ptr,        thetas_info.shape},
                 {rsqr_info.ptr,          rsqr_info.shape},
                 {solid_implant_info.ptr, solid_implant_info.shape},
                 {profile_info.ptr,       profile_info.shape}
                 );
}

void compute_front_mask(const np_array<uint8_t> &np_solid_implant,
        const float voxel_size,
        const matrix4x4 &Muvw,
        std::array<float,6> bbox,
        np_array<mask_type> &np_front_mask) {
    auto solid_implant_info = np_solid_implant.request();
    auto front_mask_info    = np_front_mask.request();

    return NS::compute_front_mask({solid_implant_info.ptr, solid_implant_info.shape},
            voxel_size, Muvw, bbox,
            {front_mask_info.ptr, front_mask_info.shape});
}

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

void cylinder_projection(const np_array<float>  &np_edt,  // Euclidean Distance Transform in um, should be low-resolution (will be interpolated)
               const np_bytearray     &np_Cs,  // Material classification images (probability per voxel, 0..1 -> 0..255)
               float Cs_voxel_size,           // Voxel size for Cs
               float d_min, float d_max,       // Distance shell to map to cylinder
               float theta_min, float theta_max, // Angle range (wrt cylinder center)
               const array<float,6> &bbox,     // Implant bounding box (in U'V'W'-coordinates)
               const matrix4x4 &Muvw,       // Transform from zyx (in um) to U'V'W' cylinder FoR (in um)
               np_array<float> &np_images,       // Probability-weighted volume of (class,theta,U)-voxels
               np_array<uint64_t> &np_counts       // Number of (class,theta,U)-voxels
               ) {
    auto edt_info    = np_edt.request();
    auto Cs_info     = np_Cs.request();
    auto images_info = np_images.request();
    auto counts_info = np_counts.request();

    NS::cylinder_projection({edt_info.ptr,edt_info.shape},
              {Cs_info.ptr, Cs_info.shape},
              Cs_voxel_size,d_min,d_max,theta_min,theta_max,bbox,Muvw,
              {images_info.ptr, images_info.shape},
              {counts_info.ptr, counts_info.shape});
}

}

PYBIND11_MODULE(geometry, m) {
    m.doc() = "Voxel Geometry Module"; // optional module docstring

    m.def("center_of_mass",       &python_api::center_of_mass, py::arg("np_voxels"));
    m.def("center_of_masses",     &python_api::center_of_masses, py::arg("np_voxels"), py::arg("np_cms").noconvert());
    m.def("inertia_matrix",       &python_api::inertia_matrix, py::arg("np_voxels"), py::arg("cm"));
    m.def("inertia_matrices",     &python_api::inertia_matrices, py::arg("np_voxels"), py::arg("np_cms"), py::arg("np_inertia_matrices").noconvert());
    m.def("integrate_axes",       &python_api::integrate_axes, py::arg("np_voxels"), py::arg("x0"), py::arg("v_axis"), py::arg("w_axis"), py::arg("v_min"), py::arg("w_min"), py::arg("output").noconvert());
    m.def("zero_outside_bbox",    &python_api::zero_outside_bbox, py::arg("principal_axes"), py::arg("parameter_ranges"), py::arg("cm"), py::arg("np_voxels").noconvert());
    m.def("fill_implant_mask",    &python_api::fill_implant_mask, py::arg("np_implant_mask"), py::arg("offset"), py::arg("voxel_size"), py::arg("bbox"), py::arg("r_fraction"), py::arg("Muvw"), py::arg("np_thetas").noconvert(), py::arg("np_rsqr_maxs").noconvert(), py::arg("np_solid_implant_mask").noconvert(), py::arg("np_profile").noconvert());
    m.def("fill_implant_mask_pre",&python_api::fill_implant_mask_pre, py::arg("np_mask"), py::arg("offset"), py::arg("voxel_size"), py::arg("bbox"), py::arg("Muvw"), py::arg("np_thetas").noconvert(), py::arg("np_rsqr_maxs").noconvert());
    m.def("cylinder_projection",  &python_api::cylinder_projection, py::arg("np_edt"), py::arg("np_Cs"), py::arg("Cs_voxel_size"), py::arg("d_min"), py::arg("d_max"), py::arg("theta_min"), py::arg("theta_max"), py::arg("bbox"), py::arg("Muvw"), py::arg("np_images").noconvert(), py::arg("np_counts").noconvert());
    m.def("sample_plane",         &python_api::sample_plane<uint16_t>, py::arg("np_voxels"), py::arg("voxel_size"), py::arg("cm"), py::arg("u_axis"), py::arg("v_axis"), py::arg("bbox"), py::arg("np_plano_samples").noconvert());
    m.def("sample_plane",         &python_api::sample_plane<uint8_t>, py::arg("np_voxels"), py::arg("voxel_size"), py::arg("cm"), py::arg("u_axis"), py::arg("v_axis"), py::arg("bbox"), py::arg("np_plano_samples").noconvert());
    m.def("compute_front_mask",   &python_api::compute_front_mask, py::arg("np_solid_implant"), py::arg("voxel_size"), py::arg("Muvw"), py::arg("bbox"), py::arg("np_front_mask").noconvert());
    m.def("compute_front_back_masks", &python_api::compute_front_back_masks, py::arg("mask"), py::arg("voxel_size"), py::arg("E"), py::arg("cm"), py::arg("cp"), py::arg("UVWp"), py::arg("front_mask").noconvert(), py::arg("back_mask").noconvert(), py::arg("implant_shell_mask").noconvert(), py::arg("solid_implant").noconvert());
}
