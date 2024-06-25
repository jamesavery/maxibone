#ifndef analysis_h
#define analysis_h

#include "boilerplate.hh"
#include "datatypes.hh"

namespace NS {

    void bic(const input_ndarray<bool> &voxels, const input_ndarray<uint16_t> &field, const input_ndarray<bool> &mask, uint16_t threshold, output_ndarray<float> &output);

}

#endif // analysis_h