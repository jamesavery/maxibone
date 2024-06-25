#ifndef analysis_h
#define analysis_h

#include "boilerplate.hh"
#include "datatypes.hh"

namespace NS {

    void bic(const input_ndarray<bool> &mask, const input_ndarray<uint16_t> &field, const uint16_t threshold);

}

#endif // analysis_h