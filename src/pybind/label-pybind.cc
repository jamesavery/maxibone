#include "label.cc"

namespace python_api {

    /**
     * Segment the voxels in `np_voxels` based on the probability distributions in `np_prob`.
     *
     * @param np_voxels The voxels volume to segment.
     * @param np_field The corresponding field volume.
     * @param np_prob The probability distributions for a voxel belonging to a material.
     * @param np_result The resulting segmentation.
     * @param vrange The value range of the voxels.
     * @param frange The value range of the fields.
     * @param offset The global offset of the volume. Used for out-of-core processing.
     * @param ranges The shape of the volume. Used for out-of-core processing.
     * @param verbose The verbosity level. Default is 0.
     */
    void material_prob_justonefieldthx(
            const py::array_t<voxel_type> &np_voxels,
            const py::array_t<field_type> &np_field,
            const py::array_t<prob_type> &np_prob,
            py::array_t<result_type> &np_result,
            const std::pair<voxel_type, voxel_type> &vrange,
            const std::pair<field_type, field_type> &frange,
            const std::tuple<uint64_t, uint64_t, uint64_t> &offset,
            const std::tuple<uint64_t, uint64_t, uint64_t> &ranges,
            const int verbose = 0) {
        NS::material_prob_justonefieldthx(np_voxels, np_field, np_prob, np_result, vrange, frange, offset, ranges, verbose);
    }

    /**
     * Perform Otsu thresholding on the bins in `np_bins`.
     * The result will be a threshold value for each row of `np_bins`, where the threshold is
     * the value that minimizes the intra-class variance within that row.
     * The length of `np_result` will be the same as the length of the first dimension of `np_bins`.
     *
     * @param np_bins The bins to threshold.
     * @param np_result The resulting thresholds.
     * @param step_size The step size to use when searching for the threshold. This is used to speed up the search at the cost of precision.
     */
    void otsu(
            const np_array<uint64_t> np_bins,
            np_array<uint64_t> np_result,
            uint64_t step_size) {
        NS::otsu(np_bins, np_result, step_size);
    }

}

PYBIND11_MODULE(label, m) {
    m.doc() = "Functions for applying material probabilities to voxels, based of the histograms and functions fitted to them. "; // optional module docstring

    m.def("material_prob_justonefieldthx", &python_api::material_prob_justonefieldthx, py::arg("np_voxels").noconvert(), py::arg("np_field").noconvert(), py::arg("np_prob").noconvert(), py::arg("np_result").noconvert(), py::arg("vrange"), py::arg("frange"), py::arg("offset"), py::arg("ranges"), py::arg("verbose") = 0);
    m.def("otsu", &python_api::otsu, py::arg("np_bins").noconvert(), py::arg("np_result").noconvert(), py::arg("step_size"));
}