#include <iostream>

#include <gtest/gtest.h>

#include <vector>
#include <algorithm>
#include <random>

#include <runningstats/runningstats.h>

#include "hdcalib.h"

static std::random_device rd;
static std::mt19937_64 engine(rd());
static std::normal_distribution<double> dist;

template<class T>
T evaluateSplineNaive(T const x, int const POS, int const DEG) {
    T const pos(POS);
    if (0 == DEG) {
        return (x <= pos || x > T(POS+1)) ? T(0) : T(1);
    }
    T const deg(DEG);
    return (x-pos)/deg*evaluateSplineNaive(x,POS,DEG-1) + (T(POS+DEG+1)-x)/deg*evaluateSplineNaive(x, POS+1, DEG-1);
}

template<class T>
T evaluateSplineExplicit(T const x, int const POS, int const DEG) {
    T const pos(POS);
    if (x < T(POS) || x > T(POS+DEG+1)) {
        return T(0);
    }
    if (0 == DEG) {
        return T(1);
    }
    if (1 == DEG) {
        if (x < T(POS+1)) {
            return x - T(POS);
        }
        return T(POS+2) - x;
    }
    if (2 == DEG) {
        if (x < T(POS+1)) {
            return (x - T(POS))*(x-T(POS))/2.0;
        }
    }
    T const deg(DEG);
    return (x-pos)/deg*evaluateSplineExplicit(x,POS,DEG-1) + (T(POS+DEG+1)-x)/deg*evaluateSplineExplicit(x, POS+1, DEG-1);
}

TEST(Spline, normal_vs_explicit) {
    runningstats::RunningStats errors;
    for (int DEG = 0; DEG <= 9; ++DEG) {
        double maxval = 0;
        for (int POS = -3; POS < 3+DEG; ++POS) {
            for (double t = POS-1; t <= POS+DEG+2; t += 0.01) {
                double const naive = evaluateSplineNaive(t, POS, DEG);
                double const expl = hdcalib::Calib::evaluateSpline(t, POS, DEG);
                ASSERT_NEAR(naive, expl, 1e-6);
                errors.push_unsafe(std::abs(naive - expl));
                maxval = std::max(maxval, naive);
            }
        }
        std::cout << "Max spline val for degree " << DEG << ": " << maxval << std::endl;
    }
    std::cout << "Errors: " << errors.print() << std::endl;
}


int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    std::cout << "RUN_ALL_TESTS return value: " << RUN_ALL_TESTS() << std::endl;
}
