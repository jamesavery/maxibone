/**
 * @file datatypes.hh
 * @author Carl-Johannes Johnsen (carl-johannes@di.ku.dk)
 * @brief Internal datatypes used throughout the project.
 * @version 0.1
 * @date 2024-09-16
 *
 * @copyright Copyright (c) 2024
 */
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

// TODO: Template + explicit instantiation

// Type of the masks.
typedef uint8_t mask_type;
// Type of the voxels.
typedef uint16_t voxel_type;
// Type of the fields.
typedef uint16_t field_type;
// Diffusion internal datatype.
typedef float gauss_type;
// The implementation type of reals.
typedef float real_t;
// Type of the solid implant mask.
typedef mask_type solid_implant_type;
// Type of the front mask (i.e. mask covering the bone region).
typedef mask_type front_mask_type;
// Type of the segmentation result.
typedef uint16_t result_type;
// Type of the probabilities for segmentation.
typedef float_t  prob_type;

// Shorthand for the pybind11 namespace.
namespace py = pybind11;
// Numpy array type.
template <typename T>
using np_array = py::array_t<T, py::array::c_style | py::array::forcecast>;

// Numpy mask array type.
typedef py::array_t<mask_type, py::array::c_style | py::array::forcecast> np_maskarray;
// Numpy `real` array type
typedef py::array_t<real_t,    py::array::c_style | py::array::forcecast> np_realarray;
// Numpy byte array type
typedef py::array_t<uint8_t,   py::array::c_style | py::array::forcecast> np_bytearray;

// TODO these should be captured during configuration or something similar, as they are device specific.
// Number of elements of type `T` per OpenACC block, which we currently set to ~1 GB.
template <typename T>
constexpr ssize_t acc_block_size = 1024 * 1024 * 1024 / sizeof(T);
// Number of threads available on GPU.
constexpr ssize_t gpu_threads = 16384;
// Above is the 4090, below is the 3080 10 GB
//constexpr ssize_t gpu_threads = 8704;

/**
 * Type of a `UV` plane, which has three `real_t` 3D vectors:
 *
 * - `cm` : Center of mass.
 * - `u_axis` : the vector defining the `U` axis of the plane.
 * - `v_axis` : the vector defining the `V` axis of the plane.
 */
struct plane_t {
    std::array<real_t, 3> cm, u_axis, v_axis;
};

/**
 * Datatype for a numpy input array.
 * The key difference between an input and output array is that the data pointer is a const pointer for input arrays.
 *
 * It has two entries:
 *
 * - `data` : a `T*` to the data.
 *
 * - `shape` : a vector holding the shape of the data.
 *
 * @tparam T the element type of the numpy array.
 */
template <typename T>
struct input_ndarray {
    // Pointer to the internal data.
    const T *data;
    // The shape of the data.
    const std::vector<ssize_t> shape;

    /**
     * Construct a new input ndarray object.
     *
     * @param arg_data the pointer to the data.
     * @param arg_shape the shape of the data.
     */
    input_ndarray(const T *arg_data, const std::vector<ssize_t> &arg_shape) : data(arg_data), shape(arg_shape) {}

    /**
     * Construct a new input ndarray object.
     *
     * @param arg_data the pointer to the data.
     * @param arg_shape the shape of the data.
     */
    input_ndarray(const void *arg_data, const std::vector<ssize_t> &arg_shape) : data(static_cast<const T*>(arg_data)), shape(arg_shape) {}
};

/**
 * Datatype for a numpy output array.
 * The key difference between an input and output array is that the data pointer is not a const pointer for output arrays.
 *
 * It has two entries:
 *
 * - `data` : a `T*` to the data.
 *
 * - `shape` : a vector holding the shape of the data.
 *
 * @tparam T the element type of the numpy array.
 */
template <typename T> struct output_ndarray {
    // Pointer to the internal data.
    T *data;
    // The shape of the data.
    const std::vector<ssize_t> shape;

    /**
     * Construct a new output ndarray object.
     *
     * @param arg_data the pointer to the data.
     * @param arg_shape the shape of the data.
     */
    output_ndarray(T *arg_data, const std::vector<ssize_t> &arg_shape) : data(arg_data), shape(arg_shape) {}

    /**
     * Construct a new output ndarray object.
     *
     * @param arg_data the pointer to the data.
     * @param arg_shape the shape of the data.
     */
    output_ndarray(void *arg_data, const std::vector<ssize_t> &arg_shape) : data(static_cast<T*>(arg_data)), shape(arg_shape) {}
};

// A four-by-four matrix of reals.
typedef std::array<real_t,16> matrix4x4;
// A vector of size 4 of reals.
typedef std::array<real_t,4>  vector4;
// A three-by-three matrix of reals.
typedef std::array<real_t,9>  matrix3x3;
// A vector of size 4 of reals.
typedef std::array<real_t,3>  vector3;

/**
 * Struct for holding the shape of a 3D array.
 *
 * It has three members:
 *
 * - `z` : the size of the z-axis.
 *
 * - `y` : the size of the y-axis.
 *
 * - `x` : the size of the x-axis.
 */
typedef struct {
    int64_t z, y, x;
} shape_t;

/**
 * Struct for holding a 3D index.
 *
 * It has three members:
 *
 * - `z` : the z coordinate.
 *
 * - `y` : the y coordinate.
 *
 * - `x` : the x coordinate.
 */
typedef struct {
    int64_t z, y, x;
} idx3d;

/**
 * Struct for holding a 3D range.
 *
 * It has 6 members:
 *
 * - `z_start` : the start of the z axis span.
 *
 * - `z_end` : the end of the z axis span.
 *
 * - `y_start` : the start of the y axis span.
 *
 * - `y_end` : the end of the y axis span.
 *
 * - `x_start` : the start of the x axis span.
 *
 * - `x_end` : the end of the x axis span.
 *
 */
struct idx3drange {
    int64_t z_start, z_end, y_start, y_end, x_start, x_end;
};

/**
 * Struct for accessing the raw bits of a 32-bit float.
 *
 * It has two members:
 *
 * - `f` : the floating point representation of the raw bytes.
 *
 * - `i` : the int32 representation of the raw bytes.
 */
typedef struct {
    union {
        float f;
        int32_t i;
    };
} raw32_t;

// A gigabyte of the `voxel_type`.
#define GB_VOXEL ((1024 / sizeof(voxel_type)) * 1024 * 1024)

/**
 * Datatype for describing the mapping between two vectors of labels. A mapping from `a` to `b` describes which labels in `b` that are neighbours to a given label in `a`.
 * It is intended to be of size `n_labels+1` as 0 is intended as background.
 */
typedef std::vector<std::unordered_set<int64_t>> mapping_t;

// Variable that enables debug printing - aka. VERY VERBOSE!
constexpr bool DEBUG = false;

// Variable that enables profiling - aka. internal timing measuring and reporting/printing.
constexpr bool PROFILE = false;

#endif // datatypes_h