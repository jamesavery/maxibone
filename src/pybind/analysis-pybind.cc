#include "analysis.cc"

namespace py = pybind11;

namespace python_api {

    void bic_wrapper(const py::array_t<bool> &mask, const py::array_t<uint16_t> &field, const uint16_t threshold, py::array_t<float> &output) {
        auto mask_info = mask.request();
        auto field_info = field.request();
        auto output_info = output.request();

        output_ndarray<float> packed_output = { output_info.ptr, output_info.shape };

        NS::bic({mask_info.ptr, mask_info.shape}, {field_info.ptr, field_info.shape}, threshold, packed_output);
    }

}

PYBIND11_MODULE(analysis, m) {
    m.doc() = "Various bone analysis functions."; // optional module docstring

    m.def("bic", &python_api::bic_wrapper, py::arg("mask").noconvert(), py::arg("field").noconvert(), py::arg("threshold"), py::arg("output").noconvert());
}