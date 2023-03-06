#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using namespace std;
namespace py = pybind11;

#include "io.cc"

namespace python_api {

template <typename T>
void load_slice(py::array_t<T> &np_data, const string filename,
                const tuple<uint64_t, uint64_t, uint64_t> offset,
                const tuple<uint64_t, uint64_t, uint64_t> shape) {
    auto data_info = np_data.request();
    T *data = static_cast<T*>(data_info.ptr);
    auto [Nz, Ny, Nx] = shape;
    auto [oz, oy, ox] = offset;
    uint64_t flat_offset = oz*Ny*Nx + oy*Nx + ox;
    NS::load_contiguous_slice<T>(data, filename, flat_offset, data_info.size);
}

template <typename T>
void write_slice(const py::array_t<T> &np_data,
        const string filename,
        const tuple<uint64_t, uint64_t, uint64_t> offset,
        const tuple<uint64_t, uint64_t, uint64_t> shape) {
    auto data_info = np_data.request();
    const T *data = static_cast<const T*>(data_info.ptr);
    auto [Nz, Ny, Nx] = shape;
    auto [oz, oy, ox] = offset;
    uint64_t flat_offset = oz*Ny*Nx + oy*Nx + ox;
    NS::write_contiguous_slice<T>(data, filename, flat_offset, data_info.size);
}

}

PYBIND11_MODULE(io, m) {
    m.doc() = "I/O functions for handling flat binary format files."; // optional module docstring
    m.def("load_slice", &python_api::load_slice<uint8_t>);
    m.def("load_slice", &python_api::load_slice<uint16_t>);
    m.def("load_slice", &python_api::load_slice<uint32_t>);
    m.def("load_slice", &python_api::load_slice<uint64_t>);
    m.def("load_slice", &python_api::load_slice<float>);
    m.def("load_slice", &python_api::load_slice<double>);

    m.def("write_slice", &python_api::write_slice<uint8_t>);
    m.def("write_slice", &python_api::write_slice<uint16_t>);
    m.def("write_slice", &python_api::write_slice<uint32_t>);
    m.def("write_slice", &python_api::write_slice<uint64_t>);
    m.def("write_slice", &python_api::write_slice<float>);
    m.def("write_slice", &python_api::write_slice<double>);
}