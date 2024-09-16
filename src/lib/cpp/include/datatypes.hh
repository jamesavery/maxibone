#ifndef datatypes_h
#define datatypes_h

#ifdef _OPENACC
//#warning "Using GPU"
#define NS gpu
#elif defined _OPENMP
//#warning "Using OpenMP"
#define NS cpu_par
#else
//#warning "Using sequential"
#define NS cpu_seq
#endif

#include <array>
#include <chrono>
#include <iostream>
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
typedef mask_type solid_implant_type;
typedef mask_type front_mask_type;

namespace py = pybind11;
template <typename T>
using np_array = py::array_t<T, py::array::c_style | py::array::forcecast>;

typedef py::array_t<mask_type, py::array::c_style | py::array::forcecast> np_maskarray;
typedef py::array_t<real_t,    py::array::c_style | py::array::forcecast> np_realarray;
typedef py::array_t<uint8_t,   py::array::c_style | py::array::forcecast> np_bytearray;

template <typename T>
constexpr ssize_t acc_block_size = 1024 * 1024 * 1024 / sizeof(T); // 1 GB
constexpr ssize_t gpu_threads = 16384*4; // 4090

struct plane_t {
  std::array<real_t,3> cm, u_axis, v_axis;
};

template <typename T> struct input_ndarray {
  const T *data;
  const std::vector<ssize_t> shape;

  input_ndarray(const T *arg_data, const std::vector<ssize_t> &arg_shape): data(arg_data), shape(arg_shape) {}
  input_ndarray(const void *arg_data, const std::vector<ssize_t> &arg_shape): data(static_cast<const T*>(arg_data)), shape(arg_shape) {}
};

template <typename T> struct output_ndarray {
  T *data;
  const std::vector<ssize_t> shape;

  output_ndarray(T *arg_data, const std::vector<ssize_t> &arg_shape): data(arg_data), shape(arg_shape) {}
  output_ndarray(void *arg_data, const std::vector<ssize_t> &arg_shape): data(static_cast<T*>(arg_data)), shape(arg_shape) {}
};

typedef std::array<real_t,16> matrix4x4;
typedef std::array<real_t,4>  vector4;
typedef std::array<real_t,9>  matrix3x3;
typedef std::array<real_t,3>  vector3;

// Struct for holding the shape of a 3D array
typedef struct {
    int64_t z, y, x;
} shape_t;

// Struct for a 3d index
typedef struct {
    int64_t z, y, x;
} idx3d;

// Struct for a 3d index range
struct idx3drange {
    int64_t z_start, z_end, y_start, y_end, x_start, x_end;
};

// Struct for accessing the raw bits of a 32-bit float
typedef struct {
    union {
        float f;
        int32_t i;
    };
} raw32_t;

#define GB_VOXEL ((1024 / sizeof(voxel_type)) * 1024 * 1024)

typedef std::vector<std::unordered_set<int64_t>> mapping_t;

constexpr bool
    DEBUG = false,
    PROFILE = false;

#endif