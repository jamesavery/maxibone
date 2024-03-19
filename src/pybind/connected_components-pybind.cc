#include "connected_components.cc"

namespace python_api {

    int64_t connected_components(const std::string &base_path, np_array<int64_t> &py_n_labels, const std::tuple<int64_t, int64_t, int64_t> &py_total_shape, const std::tuple<int64_t, int64_t, int64_t> &py_global_shape, const bool verbose = false) {
        auto n_labels_info = py_n_labels.request();
        int64_t *n_labels = static_cast<int64_t*>(n_labels_info.ptr);

        std::vector<int64_t> n_labels_vec(n_labels, n_labels + n_labels_info.size);

        const idx3d
            total_shape = {std::get<0>(py_total_shape), std::get<1>(py_total_shape), std::get<2>(py_total_shape)},
            global_shape = {std::get<0>(py_global_shape), std::get<1>(py_global_shape), std::get<2>(py_global_shape)};

        auto renamings = NS::connected_components(base_path, n_labels_vec, total_shape, global_shape, verbose);
        return NS::apply_renamings(base_path, n_labels_vec, global_shape, renamings, verbose);
    }

    void largest_connected_component(np_array<bool> &result, const std::string &base_path, np_array<int64_t> &py_n_labels, const std::tuple<int64_t, int64_t, int64_t> &py_total_shape, const std::tuple<int64_t, int64_t, int64_t> &py_global_shape, const bool verbose = false) {
        auto n_labels_info = py_n_labels.request();
        int64_t *n_labels = static_cast<int64_t*>(n_labels_info.ptr);

        std::vector<int64_t> n_labels_vec(n_labels, n_labels + n_labels_info.size);

        auto result_info = result.request();
        bool *result_data = static_cast<bool*>(result_info.ptr);

        const idx3d
            total_shape = {std::get<0>(py_total_shape), std::get<1>(py_total_shape), std::get<2>(py_total_shape)},
            global_shape = {std::get<0>(py_global_shape), std::get<1>(py_global_shape), std::get<2>(py_global_shape)};

        auto renamings = NS::connected_components(base_path, n_labels_vec, total_shape, global_shape, verbose);
        int64_t largest = NS::largest_component(base_path, renamings, n_labels_vec.size(), total_shape, global_shape, verbose);
        NS::filter_largest(base_path, result_data, renamings, largest, total_shape, global_shape, verbose);
    }

}

PYBIND11_MODULE(connected_components, m) {
    m.doc() = "Connected Components"; // optional module docstring

    m.def("connected_components", &python_api::connected_components, py::arg("base_path"), py::arg("np_n_labels"), py::arg("total_shape"), py::arg("global_shape"), py::arg("verbose") = false);
    m.def("largest_connected_component", &python_api::largest_connected_component, py::arg("result").noconvert(), py::arg("base_path"), py::arg("np_n_labels"), py::arg("total_shape"), py::arg("global_shape"), py::arg("verbose") = false);
}