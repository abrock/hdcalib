#include <iostream>

#include <gtest/gtest.h>

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <algorithm>
#include <random>

#include <ceres/ceres.h>

#include "hdmarker.hpp"
#include "hdcalib.h"

std::random_device rd;
std::default_random_engine engine(rd());
std::normal_distribution<double> dist;

::testing::AssertionResult RelativeNear(const double a, const double b, double delta) {
    double const diff = std::abs(a-b);
    double const relative_diff = 2*diff/(std::abs(a) + std::abs(b));
    if (relative_diff < delta)
        return ::testing::AssertionSuccess();
    else
        return ::testing::AssertionFailure() << "The absolute difference is " << diff
                                             << ", the relative difference is " << relative_diff
                                             << " which exceeds " << delta;
}

template<class T>
T square(const T x) {
    return x*x;
}

cv::Mat create_submarker(
        int const width,
        int const height,
        double pos_x,
        double pos_y
        ) {
    cv::Mat_<uint8_t> result(height, width);
    double const sigma = (width + height)/10;
    for (int ii = 0; ii < height; ++ii) {
        for (int jj = 0; jj < width; ++jj) {
            result(ii, jj) = std::round(255*std::exp(-(square(double(jj)-pos_x) + square(double(ii) - pos_x))/square(sigma)));
        }
    }
    return result;
}

struct SubmarkerGauss {
    uint8_t value;
    int16_t ii;
    int16_t jj;

    SubmarkerGauss(uint8_t _value, int _ii, int _jj) : value(_value), ii(_ii), jj(_jj) {}

    template<class T>
    bool operator () (
            T const * const scale,
            T const * const offset,
            T const * const x,
            T const * const y,
            T const * const sigma,
            T * residual) const {
        residual[0] = -T(value) + offset[0] + scale[0] * ceres::exp(-(square(T(jj)-x[0]) + square(T(ii) - y[0]))/square(sigma[0]));
        return true;
    }
};

cv::Point2f estimate_submarker(cv::Mat_<uint8_t> const& input, cv::Point2f const initial_guess) {
    ceres::Problem problem;


    double x = initial_guess.x, y = initial_guess.y, offset = 3, scale = 0, sigma = 1;

    for (int ii = 0; ii < input.cols; ++ii) {
        for (int jj = 0; jj < input.rows; ++jj) {
            ceres::CostFunction* cost_function =
                   new ceres::AutoDiffCostFunction<SubmarkerGauss, 1, 1, 1, 1, 1, 1>(
                       new SubmarkerGauss(input(ii, jj), ii, jj));
            problem.AddResidualBlock(cost_function, nullptr, &scale, &offset, &x, &y, &sigma);
        }
    }
    ceres::Solver::Options options;
    options.max_num_iterations = 25;
    options.linear_solver_type = ceres::DENSE_QR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    return cv::Point2f(x, y);
}

TEST(estimate_submarker, no_noise) {



}

int main(int argc, char** argv)
{

    cv::Mat_<uint8_t> sub = create_submarker(32, 32, 16, 16);
    cv::imwrite("submarker.png", sub);

    cv::Point2f initial_guess(15,18);
    estimate_submarker(sub, initial_guess);

    testing::InitGoogleTest(&argc, argv);
    std::cout << "RUN_ALL_TESTS return value: " << RUN_ALL_TESTS() << std::endl;

}
