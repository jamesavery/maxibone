/**
 * @file analysis-pybind.cc
 * Python bindings for the analysis functions.
 */
#include "analysis.cc"

namespace py = pybind11;

namespace python_api {

    /**
     * Python wrapper for the C++ function that computes the Bone Implant Contact (BIC) metric for each layer along the first (z) axis. The BIC metric is defined as the ratio of the number of voxels that are both close to the implant surface and classified as soft tissue to the total number of voxels that are close to the implant surface. The distance from the implant surface is determined by the `field` parameter and the `threshold` parameter determines the distance from the implant surface to consider being "close". The `mask` parameter is used to mask out regions of the volume that should not be considered.
     *
     * This function assumes that the output array is already allocated and has the correct shape.
     *
     * @param voxels The segmented voxels, where 1 represents soft tissue and 0 represents everything else (including background). It has shape `(Nz, Ny, Nx)`.
     * @param field The field (either gaussian, EDT or combined). It has shape `(fNz, fNy, fNx)`, where `fNz = Nz / field_scale`, `fNy = Ny / field_scale` and `fNx = Nx / field_scale` and `field_scale >= scale >= 1`.
     * @param mask A binary mask indicating which part of the volume to consider. It has shape `(mNz, mNy, mNx)`, where `mNz = Nz / mask_scale`, `mNy = Ny / mask_scale` and `mNx = Nx / mask_scale` and `mask_scale >= field_scale >= scale >= 1`.
     * @param threshold The threshold for the field. This is the parameter that chooses the distance from the implant surface to consider.
     * @param output The output array. It has shape `(Nz,)`.
     *
     * @return `void`. The result is stored in the `output` array, which is given as a parameter.
     */
    void bic_wrapper(const py::array_t<bool> &voxels, const py::array_t<uint16_t> &field, const py::array_t<bool> &mask, const uint16_t threshold, py::array_t<float> &output) {
        // Get the buffer information
        auto voxels_info = voxels.request();
        auto field_info = field.request();
        auto mask_info = mask.request();
        auto output_info = output.request();

        // Assert that the shapes are divisable by each other
        assert (voxels_info.shape[0] % field_info.shape[0] == 0);
        assert (voxels_info.shape[1] % field_info.shape[1] == 0);
        assert (voxels_info.shape[2] % field_info.shape[2] == 0);
        assert (voxels_info.shape[0] %  mask_info.shape[0] == 0);
        assert (voxels_info.shape[1] %  mask_info.shape[1] == 0);
        assert (voxels_info.shape[2] %  mask_info.shape[2] == 0);

        // Create the input and output arrays
        input_ndarray<bool> packed_voxels = { voxels_info.ptr, voxels_info.shape };
        input_ndarray<uint16_t> packed_field = { field_info.ptr, field_info.shape };
        input_ndarray<bool> packed_mask = { mask_info.ptr, mask_info.shape };
        output_ndarray<float> packed_output = { output_info.ptr, output_info.shape };

        // Call the function
        NS::bic(packed_voxels, packed_field, packed_mask, threshold, packed_output);
    }

}

// Auto-generated Python docstring: "pybind11-mkdoc pybind/analysis-pybind.cc"
static const char *__doc_python_api_bic_wrapper =
R"doc(Python wrapper for the C++ function that computes the Bone Implant
Contact (BIC) metric for each layer along the first (z) axis. The BIC
metric is defined as the ratio of the number of voxels that are both
close to the implant surface and classified as soft tissue to the
total number of voxels that are close to the implant surface. The
distance from the implant surface is determined by the `field`
parameter and the `threshold` parameter determines the distance from
the implant surface to consider being "close". The `mask` parameter is
used to mask out regions of the volume that should not be considered.

This function assumes that the output array is already allocated and
has the correct shape.

Parameter ``voxels``:
    The segmented voxels, where 1 represents soft tissue and 0
    represents everything else (including background). It has shape
    `(Nz, Ny, Nx)`.

Parameter ``field``:
    The field (either gaussian, EDT or combined). It has shape `(fNz,
    fNy, fNx)`, where `fNz = Nz / field_scale`, `fNy = Ny /
    field_scale` and `fNx = Nx / field_scale` and `field_scale >=
    scale >= 1`.

Parameter ``mask``:
    A binary mask indicating which part of the volume to consider. It
    has shape `(mNz, mNy, mNx)`, where `mNz = Nz / mask_scale`, `mNy =
    Ny / mask_scale` and `mNx = Nx / mask_scale` and `mask_scale >=
    field_scale >= scale >= 1`.

Parameter ``threshold``:
    The threshold for the field. This is the parameter that chooses
    the distance from the implant surface to consider.

Parameter ``output``:
    The output array. It has shape `(Nz,)`.

Returns:
    `void`. The result is stored in the `output` array, which is given
    as a parameter.)doc";

PYBIND11_MODULE(analysis, m) {
    m.doc() = "Various bone analysis functions."; // optional module docstring

    m.def("bic", &python_api::bic_wrapper, py::arg("voxels").noconvert(), py::arg("field").noconvert(), py::arg("mask").noconvert(), py::arg("threshold"), py::arg("output").noconvert(), __doc_python_api_bic_wrapper);
}