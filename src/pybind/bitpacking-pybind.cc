#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using namespace std;
namespace py = pybind11;

#include "bitpacking.cc"
#include "datatypes.hh"

namespace python_api {

    template <typename T>
    void encode(const py::array_t<uint8_t> &np_mask, py::array_t<T> &np_packed) {
        constexpr uint64_t T_bits = sizeof(T)*8;

        auto mask_info = np_mask.request();
        auto packed_info = np_packed.request();

        assert(packed_info.size * T_bits  >= (uint64_t) mask_info.size);

        const uint8_t *mask = static_cast<const uint8_t*>(mask_info.ptr);
        T *packed = static_cast<T*>(packed_info.ptr);

        NS::encode<T>(mask, mask_info.size, packed);
    }

    template <typename T>
    void decode(const py::array_t<T> &np_packed, py::array_t<uint8_t> &np_mask) {
        constexpr uint64_t T_bits = sizeof(T)*8;

        auto packed_info = np_packed.request();
        auto mask_info = np_mask.request();

        assert(packed_info.size * T_bits  >= (uint64_t) mask_info.size);

        const T *packed = static_cast<const T*>(packed_info.ptr);
        uint8_t *mask = static_cast<uint8_t*>(mask_info.ptr);

        NS::decode<T>(packed, mask_info.size, mask);
    }

}

PYBIND11_MODULE(bitpacking, m) {
    m.doc() = "Bitpacking functions for encoding and decoding. A bool should only take up 1 bit."; // optional module docstring

    m.def("encode", &python_api::encode<uint8_t>, py::arg("np_mask").noconvert(), py::arg("np_packed").noconvert());
    m.def("encode", &python_api::encode<uint16_t>, py::arg("np_mask").noconvert(), py::arg("np_packed").noconvert());
    m.def("encode", &python_api::encode<uint32_t>, py::arg("np_mask").noconvert(), py::arg("np_packed").noconvert());
    m.def("encode", &python_api::encode<uint64_t>, py::arg("np_mask").noconvert(), py::arg("np_packed").noconvert());

    m.def("decode", &python_api::decode<uint8_t>, py::arg("np_packed").noconvert(), py::arg("np_mask").noconvert());
    m.def("decode", &python_api::decode<uint16_t>, py::arg("np_packed").noconvert(), py::arg("np_mask").noconvert());
    m.def("decode", &python_api::decode<uint32_t>, py::arg("np_packed").noconvert(), py::arg("np_mask").noconvert());
    m.def("decode", &python_api::decode<uint64_t>, py::arg("np_packed").noconvert(), py::arg("np_mask").noconvert());
}