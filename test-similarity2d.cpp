#include <iostream>

#include <gtest/gtest.h>

#include <vector>
#include <algorithm>
#include <random>

#include "hdcalib.h"

static std::random_device rd;
static std::mt19937_64 engine(rd());
static std::normal_distribution<double> dist;

TEST(Similarity2D, exactly_solvable) {
    size_t const num_pts = 15;
    size_t const num_tests = 1000;

    for (size_t jj = 0; jj < num_tests; ++jj) {
        using hdcalib::Similarity2D;
        Similarity2D gt, solution;
        gt.angle = dist(engine)/20;
        gt.scale = .5 + std::abs(dist(engine));
        gt.t_x = dist(engine);
        gt.t_y = dist(engine);
        for (size_t ii = 0; ii < num_pts; ++ii) {
            cv::Point2d src(dist(engine), dist(engine));
            cv::Point2d dst = gt.transform(src);
            solution.src.push_back(src);
            solution.dst.push_back(dst);
        }
        solution.runFit();
        std::cout << "angle/scale/x/y: " << solution.angle << ", " << solution.scale << ", " << solution.t_x << ", " << solution.t_y << std::endl;
        ASSERT_NEAR(solution.angle, gt.angle, 1e-10);
        ASSERT_NEAR(solution.scale, gt.scale, 1e-10);
        ASSERT_NEAR(solution.t_x, gt.t_x, 1e-10);
        ASSERT_NEAR(solution.t_y, gt.t_y, 1e-10);
    }
}


int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    std::cout << "RUN_ALL_TESTS return value: " << RUN_ALL_TESTS() << std::endl;
}
