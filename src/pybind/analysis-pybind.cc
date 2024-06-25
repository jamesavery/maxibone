#include "analysis.cc"

namespace py = pybind11;

namespace python_api {

    float bic_wrapper(const py::array_t<bool> &mask, const py::array_t<uint16_t> &field, const uint16_t threshold) {
        auto mask_info = mask.request();
        auto field_info = field.request();

        return NS::bic({mask_info.ptr, mask_info.shape}, {field_info.ptr, field_info.shape}, threshold);
    }

}

PYBIND11_MODULE(analysis, m) {
   m.doc() = "Various bone analysis functions."; // optional module docstring
                                                 //
    m.def("bic", &python_api::bic_wrapper, py::arg("mask").noconvert(), py::arg("field").noconvert(), py::arg("threshold"));
}