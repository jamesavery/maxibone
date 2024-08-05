//#define _OPENACC
#include "datatypes.hh"
#include "boilerplate.hh"

namespace NS {

    inline void bincount(const input_ndarray<uint64_t> &src, output_ndarray<uint64_t> &dst) {
        UNPACK_NUMPY(src);
        UNPACK_NUMPY(dst);

        PARALLEL_TERM()
        for (int64_t flat_index = 0; flat_index < src_length; flat_index++) {
            ATOMIC()
            dst.data[src.data[flat_index]]++;
        }
    }

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

    // Assumes `allowed` is sorted
    template <typename T>
    inline void where_in(output_ndarray<T> &src, const input_ndarray<T> &allowed) {
        UNPACK_NUMPY(src);
        UNPACK_NUMPY(allowed);

        PARALLEL_TERM()
        for (int64_t i = 0; i < src_length; i++) {
            int64_t start = 0, end = allowed_Nz;
            bool found = false;
            T &elem = src.data[i];
            while (end > start) {
                int64_t mid = (start + end) / 2;
                if (elem == allowed.data[mid]) {
                    elem = 1;
                    found = true;
                    break;
                }

                if (elem < allowed.data[mid]) {
                    end = mid;
                } else {
                    start = mid + 1;
                }
            }
            if (!found) {
                elem = 0;
            }
        }
    }

}