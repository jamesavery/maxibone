#include "geometry.cc"

namespace python_api {

array<real_t, 3> center_of_mass(const np_maskarray &np_voxels){
    auto voxels_info = np_voxels.request();

    return ::center_of_mass({voxels_info.ptr, voxels_info.shape});
}

array<real_t, 9> inertia_matrix(const np_maskarray &np_voxels, array<real_t, 3>& cm){
    auto voxels_info = np_voxels.request();

    return ::inertia_matrix({voxels_info.ptr, voxels_info.shape}, cm);
}

/*

template <typename voxel_type>
void sample_plane(const np_array<voxel_type> &np_voxels,
          const real_t voxel_size, // In micrometers
          const array<real_t,3> cm,
          const array<real_t,3> u_axis,
          const array<real_t,3> v_axis,
          const array<real_t,4>  bbox,    // [umin,umax,vmin,vmax] in micrometers
          np_array<float> np_plane_samples) {
    auto voxels_info = np_voxels.request();
    auto plane_samples_info  = np_plane_samples.request();

    sample_plane<voxel_type>({voxels_info.ptr, voxels_info.shape}, voxel_size,
           cm,u_axis,v_axis,bbox,
           {plane_samples_info.ptr, plane_samples_info.shape});
}

void integrate_axes(const np_maskarray &np_voxels,
            const array<real_t,3> &x0,
            const array<real_t,3> &v_axis,
            const array<real_t,3> &w_axis,
            const real_t v_min, const real_t w_min,
            np_realarray &output) {
    auto voxels_info = np_voxels.request();
    auto output_info  = output.request();

    integrate_axes({voxels_info.ptr, voxels_info.shape},
             x0,v_axis,w_axis,
             v_min, w_min,
             {output_info.ptr, output_info.shape});
}

void zero_outside_bbox(const array<real_t,9> &principal_axes,
             const array<real_t,6> &parameter_ranges,
             const array<real_t,3> &cm, // TOOD: Med eller uden voxelsize?
             np_maskarray &np_voxels) {
    auto voxels_info = np_voxels.request();

    zero_outside_bbox(principal_axes,
              parameter_ranges,
              cm,
              {voxels_info.ptr, voxels_info.shape});
}

void fill_implant_mask(const np_maskarray implant_mask,
               float voxel_size,
               const array<float,6> &bbox,
               float r_fraction,
               const matrix4x4 &Muvw,
               np_maskarray solid_implant_mask,
               np_array<float> rsqr_maxs,
               np_array<float> profile
               ) {
    auto implant_info    = implant_mask.request(),
        solid_implant_info = solid_implant_mask.request(),
        rsqr_info          = rsqr_maxs.request(),
        profile_info       =  profile.request();

    return fill_implant_mask({implant_info.ptr,       implant_info.shape},
                 voxel_size, bbox, r_fraction, Muvw,
                 {solid_implant_info.ptr, solid_implant_info.shape},
                 {rsqr_info.ptr,          rsqr_info.shape},
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

    ::compute_front_mask({solid_implant_info.ptr, solid_implant_info.shape},
            voxel_size, Muvw, bbox,
            {front_mask_info.ptr, front_mask_info.shape});
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

    ::cylinder_projection({edt_info.ptr,edt_info.shape},
              {Cs_info.ptr, Cs_info.shape},
              Cs_voxel_size,d_min,d_max,theta_min,theta_max,bbox,Muvw,
              {images_info.ptr, images_info.shape},
              {counts_info.ptr, counts_info.shape});
}*/

}

PYBIND11_MODULE(geometry, m) {
    m.doc() = "Voxel Geometry Module"; // optional module docstring

    m.def("center_of_mass",       &python_api::center_of_mass);
    m.def("inertia_matrix",       &python_api::inertia_matrix);
    //m.def("integrate_axes",       &python_api::integrate_axes);
    //m.def("zero_outside_bbox",    &python_api::zero_outside_bbox);
    //m.def("fill_implant_mask",    &python_api::fill_implant_mask);
    //m.def("cylinder_projection",  &python_api::cylinder_projection);
    //m.def("sample_plane",         &python_api::sample_plane<uint16_t>);
    //m.def("sample_plane",         &python_api::sample_plane<uint8_t>);
    //m.def("compute_front_mask",   &python_api::compute_front_mask);
}
