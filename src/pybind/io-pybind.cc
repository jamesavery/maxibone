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

    cout <<
        Nz << " " << Ny << " " << Nx << " " <<
        oz << " " << oy << " " << ox << " " <<
        flat_offset << endl;

    NS::load_contiguous_slice<T>(data, filename, flat_offset, data_info.size);

    T checksum = (T) 0;
    for (int64_t i = 0; i < data_info.size; i++)
        checksum += data[i];
    cout << checksum << " " << sizeof(T) << endl;
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
    m.def("load_slice", &python_api::load_slice<uint8_t>, py::arg("np_data").noconvert(), py::arg("filename"), py::arg("offset"), py::arg("shape"));
    m.def("load_slice", &python_api::load_slice<uint16_t>, py::arg("np_data").noconvert(), py::arg("filename"), py::arg("offset"), py::arg("shape"));
    m.def("load_slice", &python_api::load_slice<uint32_t>, py::arg("np_data").noconvert(), py::arg("filename"), py::arg("offset"), py::arg("shape"));
    m.def("load_slice", &python_api::load_slice<uint64_t>, py::arg("np_data").noconvert(), py::arg("filename"), py::arg("offset"), py::arg("shape"));
    m.def("load_slice", &python_api::load_slice<float>, py::arg("np_data").noconvert(), py::arg("filename"), py::arg("offset"), py::arg("shape"));
    m.def("load_slice", &python_api::load_slice<double>, py::arg("np_data").noconvert(), py::arg("filename"), py::arg("offset"), py::arg("shape"));

    m.def("write_slice", &python_api::write_slice<uint8_t>, py::arg("np_data").noconvert(), py::arg("filename"), py::arg("offset"), py::arg("shape"));
    m.def("write_slice", &python_api::write_slice<uint16_t>, py::arg("np_data").noconvert(), py::arg("filename"), py::arg("offset"), py::arg("shape"));
    m.def("write_slice", &python_api::write_slice<uint32_t>, py::arg("np_data").noconvert(), py::arg("filename"), py::arg("offset"), py::arg("shape"));
    m.def("write_slice", &python_api::write_slice<uint64_t>, py::arg("np_data").noconvert(), py::arg("filename"), py::arg("offset"), py::arg("shape"));
    m.def("write_slice", &python_api::write_slice<float>, py::arg("np_data").noconvert(), py::arg("filename"), py::arg("offset"), py::arg("shape"));
    m.def("write_slice", &python_api::write_slice<double>, py::arg("np_data").noconvert(), py::arg("filename"), py::arg("offset"), py::arg("shape"));
}