#include "connected_components.cc"

namespace python_api {

    void connected_components(const std::string &base_path, np_array<int64_t> &py_n_labels, const std::tuple<int64_t, int64_t, int64_t> &py_global_shape, const bool verbose = false) {
        auto n_labels_info = py_n_labels.request();
        int64_t *n_labels = static_cast<int64_t*>(n_labels_info.ptr);

        std::vector<int64_t> n_labels_vec(n_labels, n_labels + n_labels_info.size);

        const idx3d global_shape = {std::get<0>(py_global_shape), std::get<1>(py_global_shape), std::get<2>(py_global_shape)};

        NS::connected_components(base_path, n_labels_vec, global_shape, verbose);
    }

}

PYBIND11_MODULE(connected_components, m) {
    m.doc() = "Connected Components"; // optional module docstring

    m.def("connected_components", &python_api::connected_components, py::arg("base_path"), py::arg("np_n_labels"), py::arg("global_shape"), py::arg("verbose") = false);
}