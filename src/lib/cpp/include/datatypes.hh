#ifndef datatypes_h
#define datatypes_h
#include <array>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

typedef uint8_t mask_type;	// TODO: Template + explicit instantiation
typedef uint16_t voxel_type;
typedef mask_type voxels_type;
//typedef float    field_type;
typedef uint16_t field_type;
typedef float gauss_type;
typedef float real_t;

namespace py = pybind11;
template <typename voxel_type>
using np_array = py::array_t<voxel_type, py::array::c_style | py::array::forcecast>;

typedef py::array_t<mask_type, py::array::c_style | py::array::forcecast> np_maskarray;
typedef py::array_t<real_t,    py::array::c_style | py::array::forcecast> np_realarray;
typedef py::array_t<uint8_t,   py::array::c_style | py::array::forcecast> np_bytearray;

template <typename T>
constexpr ssize_t acc_block_size = 1024 * 1024 * 1024 / sizeof(T); // 1 GB

struct plane_t {
  array<real_t,3> cm, u_axis, v_axis;
};

template <typename T> struct input_ndarray {
  const T *data;
  const vector<ssize_t> shape;

  input_ndarray(const T *arg_data, const vector<ssize_t> &arg_shape): data(arg_data), shape(arg_shape) {}
  input_ndarray(const void *arg_data, const vector<ssize_t> &arg_shape): data(static_cast<const T*>(arg_data)), shape(arg_shape) {}
};

template <typename T> struct output_ndarray {
  T *data;
  const vector<ssize_t> shape;

  output_ndarray(T *arg_data, const vector<ssize_t> &arg_shape): data(arg_data), shape(arg_shape) {}
  output_ndarray(void *arg_data, const vector<ssize_t> &arg_shape): data(static_cast<T*>(arg_data)), shape(arg_shape) {}
};

typedef std::array<real_t,16> matrix4x4;
typedef std::array<real_t,4>  vector4;
typedef std::array<real_t,9>  matrix3x3;
typedef std::array<real_t,3>  vector3;

#endif