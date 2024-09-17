#include "label.cc"

namespace python_api {

    void material_prob_justonefieldthx(
            const py::array_t<voxel_type> &np_voxels,
            const py::array_t<field_type> &np_field,
            const py::array_t<prob_type> &np_prob,
            py::array_t<result_type> &np_result,
            const std::pair<voxel_type, voxel_type> &vrange,
            const std::pair<field_type, field_type> &frange,
            const std::tuple<uint64_t, uint64_t, uint64_t> &offset,
            const std::tuple<uint64_t, uint64_t, uint64_t> &ranges) {
        NS::material_prob_justonefieldthx(np_voxels, np_field, np_prob, np_result, vrange, frange, offset, ranges);
    }

    void otsu(
            const np_array<uint64_t> np_bins,
            np_array<uint64_t> np_result,
            uint64_t step_size) {
        NS::otsu(np_bins, np_result, step_size);
    }

}

PYBIND11_MODULE(label, m) {
    m.doc() = "Functions for applying material probabilities to voxels, based of the histograms and functions fitted to them. "; // optional module docstring

    m.def("material_prob_justonefieldthx", &python_api::material_prob_justonefieldthx, py::arg("np_voxels").noconvert(), py::arg("np_field").noconvert(), py::arg("np_prob").noconvert(), py::arg("np_result").noconvert(), py::arg("vrange"), py::arg("frange"), py::arg("offset"), py::arg("ranges"));
    m.def("otsu", &python_api::otsu, py::arg("np_bins").noconvert(), py::arg("np_result").noconvert(), py::arg("step_size"));
}