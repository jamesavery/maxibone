/**
 * @file general.hh
 * @author Carl-Johannes Johnsen (carl-johannes@di.ku.dk)
 * @brief Generic functions that can be used in a variety of contexts. Mostly parallel implementations of common numpy / scipy functions.
 * @version 0.1
 * @date 2024-09-16
 *
 * @copyright Copyright (c) 2024
 */
#ifndef general_h
#define general_h

#include "datatypes.hh"
#include "boilerplate.hh"

#include <omp.h>

namespace NS {

    /**
     * Bincount. Counts the number of occurrences of each value in an array of non-negative integers.
     * It is assumed that the output is pre-allocated and zeroed.
     *
     * @param src The input array containing the non-negative integers.
     * @param dst The output array containing the counts.
     * @tparam T The datatype of the input array.
     */
    template <typename T, typename U>
    inline void bincount(const input_ndarray<T> &src, output_ndarray<U> &dst) {
        UNPACK_NUMPY(src);
        UNPACK_NUMPY(dst);

        const T *src_data = src.data;
        U *dst_data = dst.data;

        #ifdef _OPENMP
        U *local_dsts[omp_get_max_threads()];

        #pragma omp parallel
        {
            U *local_dst = (U *) calloc(dst_length, sizeof(U));
            local_dsts[omp_get_thread_num()] = local_dst;

            #pragma omp for schedule(static)
            for (int64_t flat_index = 0; flat_index < src_length; flat_index++) {
                local_dst[src_data[flat_index]]++;
            }
        }

        for (int64_t i = 0; i < dst_length; i++) {
            for (int64_t j = 0; j < omp_get_max_threads(); j++) {
                dst_data[i] += local_dsts[j][i];
            }
        }

        for (int64_t i = 0; i < omp_get_max_threads(); i++) {
            free(local_dsts[i]);
        }
        #else
        for (int64_t flat_index = 0; flat_index < src_length; flat_index++) {
            dst_data[src_data[flat_index]]++;
        }
        #endif
    }

    /**
     * Finds the minimum and maximum values in an array.
     *
     * @param in the input array.
     * @param vmin the reference to where the minimum value should be stored.
     * @param vmax the reference to where the maximum value should be stored.
     * @tparam T the internal datatype of the input array.
     */
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

    /**
     * Normalized conversion between datatypes. The output will be between the minimum and maximum values that the type `U` can represent.
     *
     * @param in the input array.
     * @param out the output array.
     * @param vmin the minimum value of the input array.
     * @param vmax the maximum value of the input array.
     * @tparam T internal datatype of the input array.
     * @tparam U internal datatype of the output array.
     */
    template <typename T, typename U>
    inline void normalized_convert(const input_ndarray<T> &in, output_ndarray<U> &out, const T vmin, const T vmax) {
        UNPACK_NUMPY(in);
        UNPACK_NUMPY(out);
        BLOCK_BEGIN_WITH_OUTPUT_TU(in, out,); {
            out_buffer[flat_index] = (U) (((float)in_buffer[flat_index] - (float)vmin) / ((float)vmax - (float)vmin) * (float)std::numeric_limits<U>::max());
        } BLOCK_END_WITH_OUTPUT_TU();
    }

    /**
     * Normalized conversion between datatypes. The output will be between the minimum and maximum values that the type `U` can represent.
     * This overload differs from the other in that it calculates the minimum and maximum values of the input array.
     *
     * @param in the input array.
     * @param out the output array.
     * @tparam T the internal datatype of the input array.
     * @tparam U the internal datatype of the output array.
     */
    template <typename T, typename U>
    inline void normalized_convert(const input_ndarray<T> &in, output_ndarray<U> &out) {
        T vmin, vmax;
        min_max(in, vmin, vmax);
        normalized_convert(in, out, vmin, vmax);
    }

    /**
     * Filters the input array `src` such that only elements that are in the `allowed` array are kept.
     * The `allowed` array is assumed to be sorted as it uses a binary search to find the elements.
     *
     * @param src the input array.
     * @param allowed the array containing the allowed values.
     * @tparam T The internal datatype of the arrays.
     */
    template <typename T>
    inline void where_in(output_ndarray<T> &src, const input_ndarray<T> &allowed) {
        UNPACK_NUMPY(src);
        UNPACK_NUMPY(allowed);

        PRAGMA(PARALLEL_TERM)
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

} // namespace NS

#endif // general_h