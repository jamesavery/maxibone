#include "analysis.cc"

namespace py = pybind11;

namespace python_api {

    void bic_wrapper(const py::array_t<bool> &voxels, const py::array_t<uint16_t> &field, const py::array_t<bool> &mask, const uint16_t threshold, py::array_t<float> &output) {
        auto voxels_info = voxels.request();
        auto field_info = field.request();
        auto mask_info = mask.request();
        auto output_info = output.request();

        // Assert that the shapes are divisable by each other
        assert (voxels_info.shape[0] % field_info.shape[0] == 0);
        assert (voxels_info.shape[1] % field_info.shape[1] == 0);
        assert (voxels_info.shape[2] % field_info.shape[2] == 0);
        assert (voxels_info.shape[0] % mask_info.shape[0] == 0);
        assert (voxels_info.shape[1] % mask_info.shape[1] == 0);
        assert (voxels_info.shape[2] % mask_info.shape[2] == 0);

        output_ndarray<float> packed_output = { output_info.ptr, output_info.shape };

        NS::bic({voxels_info.ptr, voxels_info.shape}, {field_info.ptr, field_info.shape}, {mask_info.ptr, mask_info.shape}, threshold, packed_output);
    }

}

PYBIND11_MODULE(analysis, m) {
    m.doc() = "Various bone analysis functions."; // optional module docstring

    m.def("bic", &python_api::bic_wrapper, py::arg("voxels").noconvert(), py::arg("field").noconvert(), py::arg("mask").noconvert(), py::arg("threshold"), py::arg("output").noconvert());
}