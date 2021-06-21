#ifndef LIBDOWNSCALE_H
#define LIBDOWNSCALE_H

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

namespace hdcalib {

template<class T>
cv::Mat_<T> crop(cv::Mat_<T> const & img, size_t const x, size_t const y);

template<class T>
cv::Mat_<T> downscale(cv::Mat_<T> const & img, size_t const factor);


} // namespace hdcalib

#endif // LIBDOWNSCALE_H
