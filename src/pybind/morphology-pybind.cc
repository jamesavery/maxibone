#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using namespace std;
namespace py = pybind11;

#include "morphology.cc"
#include "datatypes.hh"

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

    morphology_3d_sphere<Op, neutral>(voxels, radius, N, strides, result);
}

PYBIND11_MODULE(morphology, m) {
    m.doc() = "Morphology operations."; // optional module docstring
    m.def("dilate_3d_sphere", &morphology_3d_sphere_wrapper<std::bit_or<mask_type>, false>, py::arg("np_voxels"), py::arg("radius"), py::arg("np_result").noconvert());
    m.def("erode_3d_sphere", &morphology_3d_sphere_wrapper<std::bit_and<mask_type>, true>, py::arg("np_voxels"), py::arg("radius"), py::arg("np_result").noconvert());
}