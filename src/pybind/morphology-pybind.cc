/**
 * @file morphology.hh
 * @author Carl-Johannes Johnsen (carl-johannes@di.ku.dk)
 * @brief Python bindings for the morphology operations on 3D masks.
 * @version 0.1
 * @date 2024-09-17
 *
 * @copyright Copyright (c) 2024
 */
#include "morphology.cc"

namespace python_api {

    /**
     * 3D sphere morphology operation performed on a 3D mask.
     * If the radius is 16, a special case is called that allows for more optimizations by the compiler.
     *
     * @param np_voxels The input mask.
     * @param radius The radius of the sphere.
     * @param np_result The output mask.
     * @tparam Op The operation to perform. Should be either `std::bit_and` for erosion or `std::bit_or` for dilation.
     * @tparam neutral The neutral element for the operation. Should be either `true` for erosion or `false` for dilation.
     */
    template <typename Op, bool neutral>
    void morphology_3d_sphere_wrapper(
            const py::array_t<mask_type> &np_voxels,
            const int64_t radius,
            py::array_t<mask_type> np_result) {
        auto
            voxels_info = np_voxels.request(),
            result_info = np_result.request();

        int64_t Nz = voxels_info.shape[0], Ny = voxels_info.shape[1], Nx = voxels_info.shape[2];
        int64_t N[3] = {Nz, Ny, Nx};
        int64_t strides[3] = {Ny*Nx, Nx, 1};

        const mask_type *voxels = static_cast<const mask_type*>(voxels_info.ptr);
        mask_type *result = static_cast<mask_type*>(result_info.ptr);

        if (radius == (int64_t) 16) {
            NS::morphology_3d_sphere_r16<Op, neutral>(voxels, N, strides, result);
        } else {
            NS::morphology_3d_sphere<Op, neutral>(voxels, radius, N, strides, result);
        }
    }

    /**
     * Bitpacked 3D sphere morphology operation performed on a 3D mask.
     * The operation is performed on a bitpacked mask, where each bit represents a voxel.
     *
     * See `bitpacking.hh` for more information on bitpacking.
     *
     * @param np_voxels The input mask.
     * @param radius The radius of the sphere.
     * @param np_result The output mask.
     * @tparam op(uint32_t,uint32_t) The operation to perform.
     * @tparam reduce(uint32_t,uint32_t) The reduction to perform.
     * @tparam neutral The neutral element for the operation.
     */
    template <uint32_t op(uint32_t,uint32_t), uint32_t reduce(uint32_t,uint32_t),  uint32_t neutral>
    void morphology_3d_sphere_bitpacked_wrapper(
            const py::array_t<uint32_t> &np_voxels,
            const int64_t radius,
            py::array_t<uint32_t> np_result) {
        auto
            voxels_info = np_voxels.request(),
            result_info = np_result.request();

        int64_t Nz = voxels_info.shape[0], Ny = voxels_info.shape[1], Nx = voxels_info.shape[2];
        int64_t N[3] = {Nz, Ny, Nx*32};
        int64_t strides[3] = {Ny*Nx, Nx, 1};

        const uint32_t *voxels = static_cast<const uint32_t*>(voxels_info.ptr);
        uint32_t *result = static_cast<uint32_t*>(result_info.ptr);

        NS::morphology_3d_sphere_bitpacked<op, reduce, neutral>(voxels, radius, N, strides, result);
    }

//template <typename Op, bool neutral>
//void morphology_3d_sphere_wrapper_alt(
//        const py::array_t<mask_type> &np_voxels,
//        const int64_t radius,
//        py::array_t<mask_type> np_result) {
//    auto
//        voxels_info = np_voxels.request(),
//        result_info = np_result.request();
//
//    int64_t Nz = voxels_info.shape[0], Ny = voxels_info.shape[1], Nx = voxels_info.shape[2];
//    int64_t N[3] = {Nz, Ny, Nx};
//    int64_t strides[3] = {Ny*Nx, Nx, 1};
//
//    const mask_type *voxels = static_cast<const mask_type*>(voxels_info.ptr);
//    mask_type *result = static_cast<mask_type*>(result_info.ptr);
//
//    NS::morphology_3d_sphere_alt(voxels, radius, N, strides, result);
//}

} // namespace python_api

//
// Functions for determining the operation and reduction for the bitpacked version of the morphology operation.
//

/**
 * Dilate operation. Used for the bitpacked version of the morphology operation.
 *
 * @param a The first operand.
 * @param b The second operand.
 * @return T The result of the operation.
 * @tparam T The type of the operands.
 */
template <typename T> inline T dilate_op(const T a, const T b) { return a | b; }

/**
 * Erode operation. Used for the bitpacked version of the morphology operation.
 *
 * @param a The first operand.
 * @param b The second operand.
 * @return T The result of the operation.
 * @tparam T The type of the operands.
 */
template <typename T> inline T erode_op(const T a, const T b) { return a & b; }

/**
 * Reduction function for the dilate operation in the bitpacked version of the morphology operation.
 *
 * @param row The row to reduce.
 * @param kernel The kernel to reduce with.
 * @return T The result of the reduction.
 * @tparam T The type of the operands.
 */
template <typename T> inline T dilate_reduce(const T row, const T kernel) { return (row & kernel) != 0; };

/**
 * Reduction function for the erode operation in the bitpacked version of the morphology operation.
 *
 * @param row The row to reduce.
 * @param kernel The kernel to reduce with.
 * @return T The result of the reduction.
 * @tparam T The type of the operands.
 */
template <typename T> inline T erode_reduce(const T row, const T kernel) { return kernel == (row & kernel); }

PYBIND11_MODULE(morphology, m) {
    m.doc() = "Morphology operations."; // optional module docstring
    m.def("dilate_3d_sphere", &python_api::morphology_3d_sphere_wrapper<std::bit_or<mask_type>, false>, py::arg("np_voxels"), py::arg("radius"), py::arg("np_result").noconvert());
    m.def("dilate_3d_sphere_bitpacked", &python_api::morphology_3d_sphere_bitpacked_wrapper<dilate_op, dilate_reduce, 0>, py::arg("np_voxels"), py::arg("radius"), py::arg("np_result").noconvert());
    //m.def("dilate_3d_sphere_alt", &python_api::morphology_3d_sphere_wrapper_alt<std::bit_or<mask_type>, false>, py::arg("np_voxels"), py::arg("radius"), py::arg("np_result").noconvert());
    m.def("erode_3d_sphere", &python_api::morphology_3d_sphere_wrapper<std::bit_and<mask_type>, true>, py::arg("np_voxels"), py::arg("radius"), py::arg("np_result").noconvert());
    m.def("erode_3d_sphere_bitpacked", &python_api::morphology_3d_sphere_bitpacked_wrapper<erode_op, erode_reduce, 0xFFFFFFFF>, py::arg("np_voxels"), py::arg("radius"), py::arg("np_result").noconvert());
    //m.def("erode_3d_sphere_alt", &python_api::morphology_3d_sphere_wrapper_alt<std::bit_and<mask_type>, true>, py::arg("np_voxels"), py::arg("radius"), py::arg("np_result").noconvert());
}