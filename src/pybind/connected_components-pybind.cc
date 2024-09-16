/**
 * @file connected_components-pybind.cc
 * Python bindings for connected components C++ functions.
 */
#include "connected_components.cc"

namespace python_api {

    // Assumes that the chunks are layed out flat
    int64_t merge_labeled_chunks(np_array<int64_t> &np_chunks, np_array<int64_t> &np_n_labels, const bool verbose = false) {
        auto chunks_info = np_chunks.request();
        auto n_labels_info = np_n_labels.request();

        int64_t
            n_chunks = chunks_info.shape[0],
            nz = chunks_info.shape[1],
            ny = chunks_info.shape[2],
            nx = chunks_info.shape[3];

        int64_t *chunks = static_cast<int64_t*>(chunks_info.ptr);
        int64_t *n_labels = static_cast<int64_t*>(n_labels_info.ptr);

        const idx3d chunk_shape = {nz, ny, nx};

        return NS::merge_labeled_chunks(chunks, n_chunks, n_labels, chunk_shape, n_chunks*nz, verbose);
    }

    /**
     * Compute the connected components of the labeled chunks in `base_path`, essentially merging the labels so that there exist a global labeling across all chunks.
     * The number of chunks is given by the length of `py_n_labels`.
     *
     * @param base_path The path to the directory and prefix of the labeled chunks.
     * @param py_n_labels The number of labels in each chunk.
     * @param py_total_shape The shape of the total volume.
     * @param py_global_shape The shape of a chunk.
     * @param verbose Whether to print debug information.
     */
    int64_t connected_components(const std::string &base_path, np_array<int64_t> &py_n_labels, const std::tuple<int64_t, int64_t, int64_t> &py_total_shape, const std::tuple<int64_t, int64_t, int64_t> &py_global_shape, const bool verbose = false) {
        auto n_labels_info = py_n_labels.request();
        int64_t *n_labels = static_cast<int64_t*>(n_labels_info.ptr);

        std::vector<int64_t> n_labels_vec(n_labels, n_labels + n_labels_info.size);

        const idx3d
            total_shape = {std::get<0>(py_total_shape), std::get<1>(py_total_shape), std::get<2>(py_total_shape)},
            global_shape = {std::get<0>(py_global_shape), std::get<1>(py_global_shape), std::get<2>(py_global_shape)};

        auto renamings = NS::connected_components(base_path, n_labels_vec, global_shape, verbose);
        return NS::apply_renamings(base_path, n_labels_vec, total_shape, global_shape, renamings, verbose);
    }

    /**
     * Find the largest connected component in the labeled chunks in `base_path`.
     * The number of chunks is given by the length of `py_n_labels`.
     *
     * @param result The mask to write the largest connected component to.
     * @param base_path The path to the directory and prefix of the labeled chunks.
     * @param py_n_labels The number of labels in each chunk.
     * @param py_total_shape The shape of the total volume.
     * @param py_global_shape The shape of a chunk.
     * @param verbose Whether to print debug information.
     */
    void largest_connected_component(np_array<bool> &result, const std::string &base_path, np_array<int64_t> &py_n_labels, const std::tuple<int64_t, int64_t, int64_t> &py_total_shape, const std::tuple<int64_t, int64_t, int64_t> &py_global_shape, const bool verbose = false) {
        auto n_labels_info = py_n_labels.request();
        int64_t *n_labels = static_cast<int64_t*>(n_labels_info.ptr);

        std::vector<int64_t> n_labels_vec(n_labels, n_labels + n_labels_info.size);

        auto result_info = result.request();
        bool *result_data = static_cast<bool*>(result_info.ptr);

        const idx3d
            total_shape = {std::get<0>(py_total_shape), std::get<1>(py_total_shape), std::get<2>(py_total_shape)},
            global_shape = {std::get<0>(py_global_shape), std::get<1>(py_global_shape), std::get<2>(py_global_shape)};

        auto renamings = NS::connected_components(base_path, n_labels_vec, global_shape, verbose);
        int64_t largest = NS::largest_component(base_path, renamings, n_labels_vec[0], total_shape, global_shape, verbose);
        NS::filter_largest(base_path, result_data, renamings, largest, total_shape, global_shape, verbose);
    }

}

PYBIND11_MODULE(connected_components, m) {
    m.doc() = "Connected Components"; // optional module docstring

    m.def("connected_components", &python_api::connected_components, py::arg("base_path"), py::arg("np_n_labels"), py::arg("total_shape"), py::arg("global_shape"), py::arg("verbose") = false);
    m.def("largest_connected_component", &python_api::largest_connected_component, py::arg("result").noconvert(), py::arg("base_path"), py::arg("np_n_labels"), py::arg("total_shape"), py::arg("global_shape"), py::arg("verbose") = false);
    m.def("merge_labeled_chunks", &python_api::merge_labeled_chunks, py::arg("np_chunks").noconvert(), py::arg("np_n_labels").noconvert(), py::arg("verbose") = false);
}