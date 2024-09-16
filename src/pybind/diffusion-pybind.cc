/**
 * @file diffusion-pybind.cc
 * @author Carl-Johannes Johnsen (carl-johannes@di.ku.dk)
 * @brief Python bindings for 3D Diffusion approximation using repeated applications of a 1D gaussian in each dimension.
 * @version 0.1
 * @date 2024-09-16
 *
 * @copyright Copyright (c) 2024
 */
#include "diffusion.cc"

namespace python_api {

    /**
     * In-memory 3D diffusion approximation using repeated applications of a 1D gaussian in each dimension.
     * This version assumes that the host has enough memory to hold both input, output and working memory.
     * This translates to `N.z*N.y*N.x * (sizeof(uint8_t) + 2*sizeof(float) + sizeof(uint16_t)) + kernel_size`.
     * If the GPU implementation is chosen and the  total size of the input, output and working memory exceeds the memory available on the GPU device, the out-of-core implementation is chosen.
     * Otherwise, the in-memory implementation is chosen.
     *
     * @param np_voxels The input mask.
     * @param np_kernel The 1D gaussian kernel to apply.
     * @param np_output The output array.
     * @param repititions The number of repititions to apply the kernel.
     */
    void diffusion_in_memory(np_array<uint8_t> &np_voxels, const np_array<float> &np_kernel, np_array<uint16_t> &np_output, const int64_t repititions = 1) {
        auto
            voxels_info = np_voxels.request(),
            kernel_info = np_kernel.request(),
            output_info = np_output.request();

        uint8_t *voxels = static_cast<uint8_t*>(voxels_info.ptr);
        const float *kernel = static_cast<const float*>(kernel_info.ptr);
        uint16_t *output = static_cast<uint16_t*>(output_info.ptr);

        const shape_t N = {voxels_info.shape[0], voxels_info.shape[1], voxels_info.shape[2]};
        const int64_t kernel_size = kernel_info.shape[0];

#ifdef _OPENACC
        const int64_t total_size = N.z * N.y * N.x;
        if (total_size * (sizeof(uint8_t) + (2 * sizeof(float)) + sizeof(uint16_t)) > (9 * 1e9)) { // TODO make automatic - set to ~90% of the 3080's 10 GB
        if (true) {
            const shape_t global_shape = { 128, 128, 128 }; // TODO balancing act. Larger results in less wasted compute, while smaller results in more concurrency.
            NS::diffusion_out_of_core(voxels, N, global_shape, kernel, kernel_size, repititions, output);
        } else {
#endif
            NS::diffusion_in_memory(voxels, N, kernel, kernel_size, repititions, output);
#ifdef _OPENACC
        }
#endif


    }

    /**
     * On-disk 3D diffusion approximation using repeated applications of a 1D gaussian in each dimension.
     * This version reads and writes the results to disk, as they are assumed too large to fit in memory.
     *
     * @param input_file The path to the input file.
     * @param np_kernel The 1D gaussian kernel to apply.
     * @param output_file The path to the output file.
     * @param py_total_shape The shape of the total volume.
     * @param py_global_shape The shape of each chunk to load in and out.
     * @param repititions The number of repititions to apply the kernel.
     * @param verbose Whether debug information should be printed.
     */
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