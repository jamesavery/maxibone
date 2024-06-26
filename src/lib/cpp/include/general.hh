//#define _OPENACC
#include "datatypes.hh"
#include "boilerplate.hh"

namespace NS {

    template <typename T>
    inline void min_max(const input_ndarray<T> &in, T &vmin, T &vmax) {
        UNPACK_NUMPY(in);
        vmin = std::numeric_limits<T>::max();
        vmax = std::numeric_limits<T>::min();
        BLOCK_BEGIN_T(in, reduction(min:vmin) reduction(max:vmax)); {
            vmin = std::min(vmin, in_buffer[flat_index]);
            vmax = std::max(vmax, in_buffer[flat_index]);
        } BLOCK_END_T();
    }

    template <typename T, typename U>
    inline void normalized_convert(const input_ndarray<T> &in, output_ndarray<U> &out, const T vmin, const T vmax) {
        UNPACK_NUMPY(in);
        UNPACK_NUMPY(out);
        BLOCK_BEGIN_WITH_OUTPUT_TU(in, out,); {
            out_buffer[flat_index] = (U) (((float)in_buffer[flat_index] - (float)vmin) / ((float)vmax - (float)vmin) * (float)std::numeric_limits<U>::max());
        } BLOCK_END_WITH_OUTPUT_TU();
    }

    template <typename T, typename U>
    inline void normalized_convert(const input_ndarray<T> &in, output_ndarray<U> &out) {
        T vmin, vmax;
        min_max(in, vmin, vmax);
        normalized_convert(in, out, vmin, vmax);
    }

}