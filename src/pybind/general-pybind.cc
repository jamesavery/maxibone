#include "general.hh"

namespace py = pybind11;

namespace python_api {

    void bincount(const np_array<uint64_t> &np_src, np_array<uint64_t> np_dst) {
        auto src_info = np_src.request();
        auto dst_info = np_dst.request();

        const input_ndarray<uint64_t> src = { src_info.ptr, src_info.shape };
        output_ndarray<uint64_t> dst = { dst_info.ptr, dst_info.shape };

        NS::bincount(src, dst);
    }

    template <typename T, typename U>
    void normalized_convert(const np_array<T> &np_src, np_array<U> np_dst) {
        auto src_info = np_src.request();
        auto dst_info = np_dst.request();

        const input_ndarray<T> src = { src_info.ptr, src_info.shape };
        output_ndarray<U> dst = { dst_info.ptr, dst_info.shape };

        NS::normalized_convert(src, dst);
    }

}

PYBIND11_MODULE(general, m) {
    m.doc() = "Generic functions."; // optional module docstring

    m.def("bincount", &python_api::bincount, py::arg("np_src").noconvert(), py::arg("np_dst").noconvert());
    m.def("normalized_convert", &python_api::normalized_convert<float, uint8_t>, py::arg("np_src").noconvert(), py::arg("np_dst").noconvert());
    m.def("normalized_convert", &python_api::normalized_convert<float, uint16_t>, py::arg("np_src").noconvert(), py::arg("np_dst").noconvert());
}