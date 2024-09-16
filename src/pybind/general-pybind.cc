/**
 * @file general.cc
 * @author Carl-Johannes Johnsen (carl-johannes@di.ku.dk)
 * @brief Python bindings for generic functions that can be used in a variety of contexts. Mostly parallel implementations of common numpy / scipy functions.
 * @version 0.1
 * @date 2024-09-16
 *
 * @copyright Copyright (c) 2024
 */
#include "general.hh"

namespace py = pybind11;

namespace python_api {

    /**
     * Bincount. Counts the number of occurrences of each value in an array of non-negative integers.
     * It is assumed that the output is pre-allocated and zeroed.
     *
     * @param np_src The input array containing the non-negative integers.
     * @param np_dst The output array containing the counts.
     * @tparam T The datatype of the input array.
     */
    void bincount(const np_array<uint64_t> &np_src, np_array<uint64_t> np_dst) {
        auto src_info = np_src.request();
        auto dst_info = np_dst.request();

        const input_ndarray<uint64_t> src = { src_info.ptr, src_info.shape };
        output_ndarray<uint64_t> dst = { dst_info.ptr, dst_info.shape };

        NS::bincount(src, dst);
    }

    /**
     * Normalized conversion between datatypes. The output will be between the minimum and maximum values that the type `U` can represent.
     * This overload differs from the other in that it calculates the minimum and maximum values of the input array.
     *
     * @param np_src the input array.
     * @param np_dst the output array.
     * @tparam T the internal datatype of the input array.
     * @tparam U the internal datatype of the output array.
     */
    template <typename T, typename U>
    void normalized_convert(const np_array<T> &np_src, np_array<U> np_dst) {
        auto src_info = np_src.request();
        auto dst_info = np_dst.request();

        const input_ndarray<T> src = { src_info.ptr, src_info.shape };
        output_ndarray<U> dst = { dst_info.ptr, dst_info.shape };

        NS::normalized_convert(src, dst);
    }

    /**
     * Filters the input array `src` such that only elements that are in the `allowed` array are kept.
     * The `allowed` array is assumed to be sorted as it uses a binary search to find the elements.
     *
     * @param np_src the input array.
     * @param np_allowed the array containing the allowed values.
     * @tparam T The internal datatype of the arrays.
     */
    template <typename T>
    void where_in(np_array<T> &np_src, const np_array<T> &np_allowed) {
        auto src_info = np_src.request();
        auto allowed_info = np_allowed.request();

        output_ndarray<T> src = { src_info.ptr, src_info.shape };
        const input_ndarray<T> allowed = { allowed_info.ptr, allowed_info.shape };

        NS::where_in(src, allowed);
    }

}

PYBIND11_MODULE(general, m) {
    m.doc() = "Generic functions."; // optional module docstring

    m.def("bincount", &python_api::bincount, py::arg("np_src").noconvert(), py::arg("np_dst").noconvert());
    m.def("normalized_convert", &python_api::normalized_convert<float, uint8_t>, py::arg("np_src").noconvert(), py::arg("np_dst").noconvert());
    m.def("normalized_convert", &python_api::normalized_convert<float, uint16_t>, py::arg("np_src").noconvert(), py::arg("np_dst").noconvert());
    m.def("where_in", &python_api::where_in<uint64_t>, py::arg("np_src").noconvert(), py::arg("np_allowed").noconvert());
}