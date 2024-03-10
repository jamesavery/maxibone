#ifndef histograms_h
#define histograms_h

#include "datatypes.hh"
#include <chrono>

namespace NS {

    void axis_histogram(const voxel_type __restrict__* voxels,
                        const shape_t &voxels_shape,
                        const shape_t &offset,
                        const shape_t &block_size,
                        uint64_t __restrict__* x_bins,
                        uint64_t __restrict__* y_bins,
                        uint64_t __restrict__* z_bins,
                        uint64_t __restrict__* r_bins,
                        const uint64_t voxel_bins,
                        const uint64_t Nr,
                        const std::tuple<uint64_t, uint64_t> &center,
                        const std::tuple<double, double> &vrange,
                        const bool verbose);

}

#endif