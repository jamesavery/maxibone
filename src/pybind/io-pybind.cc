/**
 * @file io-pybind.cc
 * Python bindings for the I/O functions.
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "io.cc"

namespace python_api {

    /**
     * Load a contiguous slice from a flat binary file into a numpy array.
     *
     * @param np_data The numpy array to load the data into.
     * @param filename The name of the file to load the data from.
     * @param offset The offset of the slice to load.
     * @param shape The shape of the slice to load.
     * @tparam T The type of the data to load.
     */
    template <typename T>
    void load_slice(py::array_t<T> &np_data, const std::string filename,
                    const std::tuple<uint64_t, uint64_t, uint64_t> offset,
                    const std::tuple<uint64_t, uint64_t, uint64_t> shape) {
        auto data_info = np_data.request();
        T *data = static_cast<T*>(data_info.ptr);
        auto [Nz, Ny, Nx] = shape;
        auto [oz, oy, ox] = offset;
        uint64_t flat_offset = oz*Ny*Nx + oy*Nx + ox;

        assert (data_info.size >= (int64_t) (Nz*Ny*Nx));

        NS::load_contiguous_slice<T>(data, filename, flat_offset, data_info.size);
    }

    /**
     * Write a contiguous slice from a numpy array to a flat binary file.
     *
     * @param np_data The numpy array to write the data from.
     * @param filename The name of the file to write the data to.
     * @param offset The offset of the slice to write.
     * @param shape The shape of the slice to write.
     * @tparam T The type of the data to write.
     */
    template <typename T>
    void write_slice(const py::array_t<T> &np_data,
            const std::string filename,
            const std::tuple<uint64_t, uint64_t, uint64_t> offset,
            const std::tuple<uint64_t, uint64_t, uint64_t> shape) {
        auto data_info = np_data.request();
        const T *data = static_cast<const T*>(data_info.ptr);
        auto [Nz, Ny, Nx] = shape;
        auto [oz, oy, ox] = offset;
        uint64_t flat_offset = oz*Ny*Nx + oy*Nx + ox;

        assert(data_info.size >= (int64_t) (Nz*Ny*Nx));

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