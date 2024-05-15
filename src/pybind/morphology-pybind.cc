#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using namespace std;
namespace py = pybind11;

#include "morphology.cc"
#include "datatypes.hh"

namespace python_api {

template <typename Op, bool neutral>
void morphology_3d_sphere_wrapper(
        const py::array_t<mask_type> &np_voxels,
        const int64_t radius,
        py::array_t<mask_type> np_result) {
    auto
        voxels_info = np_voxels.request(),
        result_info = np_result.request();

    int64_t Nz = voxels_info.shape[0], Ny = voxels_info.shape[1], Nx = voxels_info.shape[2];
    int64_t N[3] = {Nz, Ny, Nx};
    int64_t strides[3] = {Ny*Nx, Nx, 1};

    const mask_type *voxels = static_cast<const mask_type*>(voxels_info.ptr);
    mask_type *result = static_cast<mask_type*>(result_info.ptr);

    if (radius == (int64_t) 16) {
        //NS::morphology_3d_sphere_r16<Op, neutral>(voxels, N, strides, result);
    } else {
        NS::morphology_3d_sphere<Op, neutral>(voxels, radius, N, strides, result);
    }
}

template <typename Op, uint32_t neutral>
void morphology_3d_sphere_bitpacked(
        const uint32_t *voxels,
        const int64_t radius,
        const int64_t N[3],
        const int64_t strides[3],
        uint32_t *result);

template <typename Op, uint32_t neutral>
void morphology_3d_sphere_bitpacked_wrapper(
        const py::array_t<uint32_t> &np_voxels,
        const int64_t radius,
        py::array_t<uint32_t> np_result) {
    auto
        voxels_info = np_voxels.request(),
        result_info = np_result.request();

    int64_t Nz = voxels_info.shape[0], Ny = voxels_info.shape[1], Nx = voxels_info.shape[2];
    int64_t N[3] = {Nz, Ny, Nx*32};
    int64_t strides[3] = {Ny*Nx, Nx, 1};

    const uint32_t *voxels = static_cast<const uint32_t*>(voxels_info.ptr);
    uint32_t *result = static_cast<uint32_t*>(result_info.ptr);

    NS::morphology_3d_sphere_bitpacked<Op, neutral>(voxels, radius, N, strides, result);
}

//template <typename Op, bool neutral>
//void morphology_3d_sphere_wrapper_alt(
//        const py::array_t<mask_type> &np_voxels,
//        const int64_t radius,
//        py::array_t<mask_type> np_result) {
//    auto
//        voxels_info = np_voxels.request(),
//        result_info = np_result.request();
//
//    int64_t Nz = voxels_info.shape[0], Ny = voxels_info.shape[1], Nx = voxels_info.shape[2];
//    int64_t N[3] = {Nz, Ny, Nx};
//    int64_t strides[3] = {Ny*Nx, Nx, 1};
//
//    const mask_type *voxels = static_cast<const mask_type*>(voxels_info.ptr);
//    mask_type *result = static_cast<mask_type*>(result_info.ptr);
//
//    NS::morphology_3d_sphere_alt(voxels, radius, N, strides, result);
//}

} // namespace python_api

PYBIND11_MODULE(morphology, m) {
    m.doc() = "Morphology operations."; // optional module docstring
    m.def("dilate_3d_sphere", &python_api::morphology_3d_sphere_wrapper<std::bit_or<mask_type>, false>, py::arg("np_voxels"), py::arg("radius"), py::arg("np_result").noconvert());
    m.def("dilate_3d_sphere_bitpacked", &python_api::morphology_3d_sphere_bitpacked_wrapper<std::bit_or<uint32_t>, 0>, py::arg("np_voxels"), py::arg("radius"), py::arg("np_result").noconvert());
    //m.def("dilate_3d_sphere_alt", &python_api::morphology_3d_sphere_wrapper_alt<std::bit_or<mask_type>, false>, py::arg("np_voxels"), py::arg("radius"), py::arg("np_result").noconvert());
    m.def("erode_3d_sphere", &python_api::morphology_3d_sphere_wrapper<std::bit_and<mask_type>, true>, py::arg("np_voxels"), py::arg("radius"), py::arg("np_result").noconvert());
    //m.def("erode_3d_sphere_alt", &python_api::morphology_3d_sphere_wrapper_alt<std::bit_and<mask_type>, true>, py::arg("np_voxels"), py::arg("radius"), py::arg("np_result").noconvert());
}