#include "libdownscale.h"

#include <cmath>

#include "randutils.hpp"

namespace  {
randutils::mt19937_rng rng;
} // anonymous namespace


namespace hdcalib {


template<class T>
cv::Mat_<T> crop(const cv::Mat_<T> &img, const size_t x, const size_t y) {
    if (x >= size_t(img.cols) || y >= size_t(img.rows)) {
        return img;
    }
    cv::Rect roi(x, y, img.size().width - x, img.size().height - y);
    return img(roi);
}

double aliasingRound(double const val) {
    double const floor = std::floor(val);
    double const residual = val - floor;
    double const dice = rng.uniform(0.0, 1.0);
    if (dice > residual) {
        return floor;
    }
    return floor+1;
}

template<class T>
cv::Mat_<T> downscale(const cv::Mat_<T> &img, const size_t factor) {
    if (factor < 2) {
        return img;
    }
    cv::Mat_<T> result(img.rows/factor, img.cols/factor, 0.0);
    if (result.rows < 1 || result.cols < 1) {
        return result;
    }
    for (int ii = 0; ii < result.rows; ++ii) {
        for (int jj = 0; jj < result.cols; ++jj) {
            double sum = 0;
            for (int dii = 0; dii < int(factor); ++dii) {
                for (int djj = 0; djj < int(factor); ++djj) {
                    sum += img(factor*ii + dii, factor*jj + djj);
                }
            }
            result(ii, jj) = cv::saturate_cast<T>(aliasingRound(sum/(factor*factor)));
        }
    }
    return result;
}

template cv::Mat_<float> downscale(const cv::Mat_<float> &img, const size_t factor);
template cv::Mat_<double> downscale(const cv::Mat_<double> &img, const size_t factor);

template cv::Mat_<float> crop(const cv::Mat_<float> &img, const size_t x, const size_t y);
template cv::Mat_<double> crop(const cv::Mat_<double> &img, const size_t x, const size_t y);

} // namespace hdcalib
