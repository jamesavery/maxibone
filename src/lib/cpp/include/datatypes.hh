#ifndef datatypes_h
#define datatypes_h
#include <array>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

typedef uint8_t mask_type;	// TODO: Template + explicit instantiation
typedef uint16_t voxel_type;
//typedef float    field_type;
typedef uint16_t field_type;
typedef float gauss_type;
typedef float real_t;

namespace py = pybind11;
template <typename voxel_type>
using np_array = py::array_t<voxel_type, py::array::c_style | py::array::forcecast>;

typedef py::array_t<mask_type, py::array::c_style | py::array::forcecast> np_maskarray;
typedef py::array_t<real_t, py::array::c_style | py::array::forcecast>    np_realarray;
typedef py::array_t<uint8_t, py::array::c_style | py::array::forcecast>   np_bytearray;

constexpr ssize_t acc_block_size = 1024 * 1024 * 1024 / sizeof(mask_type); // 1 GB

struct plane_t {
  array<real_t,3> cm, u_axis, v_axis;
};

template <typename T> struct input_ndarray {
  const T *data;
  const vector<ssize_t> shape;

  input_ndarray(const T *data, const vector<ssize_t> &shape): data(data), shape(shape) {}
  input_ndarray(const void *data, const vector<ssize_t> &shape): data(static_cast<const T*>(data)), shape(shape) {}
};

template <typename T> struct output_ndarray {
  T *data;
  const vector<ssize_t> shape;

  output_ndarray(T *data, const vector<ssize_t> &shape): data(data), shape(shape) {}
  output_ndarray(void *data, const vector<ssize_t> &shape): data(static_cast<T*>(data)), shape(shape) {}
};

typedef std::array<real_t,16> matrix4x4;
typedef std::array<real_t,4>  vector4;
typedef std::array<real_t,9>  matrix3x3;
typedef std::array<real_t,3>  vector3;

#endif