#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using namespace std;
namespace py = pybind11;

#include "datatypes.hh"
#include "io.cc"

template <typename T>
void load_slice(py::array_t<T> &np_data, const string filename,
                const tuple<uint64_t, uint64_t, uint64_t> offset,
                const tuple<uint64_t, uint64_t, uint64_t> shape) {
    auto data_info = np_data.request();
    T *data = static_cast<T*>(data_info.ptr);
    auto [Nz, Ny, Nx] = shape;
    auto [oz, oy, ox] = offset;
    uint64_t flat_offset = oz*Ny*Nx + oy*Nx + ox;
    load_contiguous_slice<T>(data, filename, flat_offset, data_info.size);
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
    write_contiguous_slice<T>(data, filename, flat_offset, data_info.size);
}

PYBIND11_MODULE(io, m) {
    m.doc() = "I/O functions for handling flat binary format files."; // optional module docstring
    m.def("load_slice", &load_slice<mask_type>);
    m.def("load_slice", &load_slice<voxel_type>);
    m.def("load_slice", &load_slice<field_type>);
    m.def("load_slice", &load_slice<gauss_type>);
    m.def("load_slice", &load_slice<real_t>);

    m.def("write_slice", &write_slice<mask_type>);
    m.def("write_slice", &write_slice<voxel_type>);
    m.def("write_slice", &write_slice<field_type>);
    m.def("write_slice", &write_slice<gauss_type>);
    m.def("write_slice", &write_slice<real_t>);
}