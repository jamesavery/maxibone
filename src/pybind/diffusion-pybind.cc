#include "diffusion.cc"

namespace python_api {

    void diffusion_in_memory(const np_array<uint8_t> &np_voxels, const np_array<float> &np_kernel, np_array<uint8_t> &np_output, const int64_t repititions = 1) {
        auto
            voxels_info = np_voxels.request(),
            kernel_info = np_kernel.request(),
            output_info = np_output.request();

        const uint8_t *voxels = static_cast<const uint8_t*>(voxels_info.ptr);
        const float *kernel = static_cast<const float*>(kernel_info.ptr);
        uint8_t *output = static_cast<uint8_t*>(output_info.ptr);

        const shape_t N = {voxels_info.shape[0], voxels_info.shape[1], voxels_info.shape[2]};
        const int64_t kernel_size = kernel_info.shape[0];

        NS::diffusion_in_memory(voxels, N, kernel, kernel_size, repititions, output);
    }

    void diffusion_on_disk(const std::string &input_file, const np_array<float> &np_kernel, const std::string &output_file, const std::tuple<int64_t, int64_t, int64_t> &py_total_shape, const std::tuple<int64_t, int64_t, int64_t> &py_global_shape, const int64_t repititions, const bool verbose = false) {
        shape_t
            total_shape = {std::get<0>(py_total_shape), std::get<1>(py_total_shape), std::get<2>(py_total_shape)},
            global_shape = {std::get<0>(py_global_shape), std::get<1>(py_global_shape), std::get<2>(py_global_shape)};

        auto kernel_info = np_kernel.request();
        const float *kernel = static_cast<const float*>(kernel_info.ptr);
        const int64_t kernel_size = kernel_info.shape[0];

        NS::diffusion_on_disk(input_file, kernel, kernel_size, output_file, total_shape, global_shape, repititions, verbose);
    }

}

PYBIND11_MODULE(diffusion, m) {
    m.doc() = "Diffusion approximation using 3 1D gauss convolutions."; // optional module docstring

    m.def("diffusion", &python_api::diffusion_in_memory, py::arg("np_voxels"), py::arg("np_kernel"), py::arg("np_output").noconvert(), py::arg("repititions") = 1);
    m.def("diffusion", &python_api::diffusion_on_disk, py::arg("input_file"), py::arg("np_kernel"), py::arg("output_file"), py::arg("total_shape"), py::arg("global_shape"), py::arg("repititions") = 1, py::arg("verbose") = false);
}