#include "hdcalib.h"

namespace hdcalib {

template<class T>
T Calib::lensfunDistortionModel(T const & a, T const& b, T const& c, T const& r) {
    return (a*r*r*r + b*r*r + c*r + 1 - a - b - c)*r;
}


} // namespace hdcalib
