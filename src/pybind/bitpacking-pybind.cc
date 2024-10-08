/**
 * @file bitpacking-pybind.cc
 * Python bindings for bitpacking C++ functions.
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "bitpacking.cc"
#include "datatypes.hh"

namespace python_api {

    template <typename T>
    void encode(const py::array_t<uint8_t> &np_mask, py::array_t<T> &np_packed, const int verbose) {
        constexpr uint64_t T_bits = sizeof(T)*8;

        auto mask_info = np_mask.request();
        auto packed_info = np_packed.request();

        assert(packed_info.size * T_bits  >= (uint64_t) mask_info.size && "Packed array is too small");
        assert(mask_info.size % (T_bits*T_bits) == 0 && "Mask size must be a multiple of T_bits*T_bits");

        const uint8_t *mask = static_cast<const uint8_t*>(mask_info.ptr);
        T *packed = static_cast<T*>(packed_info.ptr);

        NS::encode(mask, mask_info.size, packed, verbose);
    }

    template <typename T>
    void decode(const py::array_t<T> &np_packed, py::array_t<uint8_t> &np_mask, const int verbose) {
        constexpr uint64_t T_bits = sizeof(T)*8;

        auto packed_info = np_packed.request();
        auto mask_info = np_mask.request();

        assert(packed_info.size * T_bits  >= (uint64_t) mask_info.size);

        const T *packed = static_cast<const T*>(packed_info.ptr);
        uint8_t *mask = static_cast<uint8_t*>(mask_info.ptr);

        NS::decode(packed, mask_info.size, mask, verbose);
    }

}

PYBIND11_MODULE(bitpacking, m) {
    m.doc() = "Bitpacking functions for encoding and decoding. A bool should only take up 1 bit. Current implementations are built around the packed datatype being uint32_t."; // optional module docstring

    // TODO Currently, the GPU implementation only supports uint32_t.

    //m.def("encode", &python_api::encode<uint8_t>, py::arg("np_mask").noconvert(), py::arg("np_packed").noconvert(), py::arg("verbose") = 0);
    //m.def("encode", &python_api::encode<uint16_t>, py::arg("np_mask").noconvert(), py::arg("np_packed").noconvert(), py::arg("verbose") = 0);
    m.def("encode", &python_api::encode<uint32_t>, py::arg("np_mask").noconvert(), py::arg("np_packed").noconvert(), py::arg("verbose") = 0);
    //m.def("encode", &python_api::encode<uint64_t>, py::arg("np_mask").noconvert(), py::arg("np_packed").noconvert(), py::arg("verbose") = 0);

    //m.def("decode", &python_api::decode<uint8_t>, py::arg("np_packed").noconvert(), py::arg("np_mask").noconvert(), py::arg("verbose") = 0);
    //m.def("decode", &python_api::decode<uint16_t>, py::arg("np_packed").noconvert(), py::arg("np_mask").noconvert(), py::arg("verbose") = 0);
    m.def("decode", &python_api::decode<uint32_t>, py::arg("np_packed").noconvert(), py::arg("np_mask").noconvert(), py::arg("verbose") = 0);
    //m.def("decode", &python_api::decode<uint64_t>, py::arg("np_packed").noconvert(), py::arg("np_mask").noconvert(), py::arg("verbose") = 0);
}