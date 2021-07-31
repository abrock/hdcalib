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

template<int NUM, int DEG>
struct SplineFunctor {
    cv::Vec2f src, dst;
    cv::Size size;
    double factor_x, factor_y;

    static const size_t n = (NUM+DEG)*(NUM+DEG);
    static const size_t n_rows = (NUM+DEG);

    SplineFunctor(cv::Vec2f const& _src, cv::Vec2f const& _dst, cv::Size const& _size) :
        src(_src),
        dst(_dst),
        size(_size),
        factor_x(double(NUM)/size.width), factor_y(double(NUM)/size.height){}

    cv::Vec2f apply(cv::Vec2f const& pt,
                    cv::Mat_<float> const& weights_x,
                    cv::Mat_<float> weights_y) const {
        cv::Vec2f result(pt);
        apply(result.val, weights_x.ptr<float>(), weights_y.ptr<float>());
        return result;
    }

    cv::Vec2f apply(cv::Vec2f const& pt,
                    std::vector<double> const& weights_x,
                    std::vector<double> weights_y) const {
        cv::Vec2f result(pt);
        apply(result.val, weights_x.data(), weights_y.data());
        return result;
    }

    template<class T>
    bool operator()(T const * const weights_x, T const * const weights_y, T * residuals) const {
        residuals[0] = T(src[0]);
        residuals[1] = T(src[1]);
        apply(residuals, weights_x, weights_y);
        residuals[0] -= T(dst[0]);
        residuals[1] -= T(dst[1]);
        return true;
    }

    template<class T, class U>
    void apply(T* pt, U const * const weights_x, U const * const weights_y) const {
        T scaled_pt[2] = {pt[0]*T(factor_x), pt[1]*T(factor_y)};
        T const dx = applySingle(scaled_pt, weights_x);
        T const dy = applySingle(scaled_pt, weights_y);
        pt[0] += dx;
        pt[1] += dy;
    }

    template<class T, class U>
    T applySingle(T const * const val, U const * const weights) const {
        T col[n_rows];
        for (size_t ii = 0; ii < n_rows; ++ii) {
            int const POS = int(ii) - DEG;
            if (val[1] > POS && val[1] < POS+DEG+1) {
                col[ii] = applyRow(val[0], &weights[ii*n_rows]);
            }
            else {
                col[ii] = 0;
            }
        }
        return applyRow(val[1], col);
    }

    template<class T, class U>
    T applyRow(T const& val, U const * const weights) const {
        T result(0);
        for (int ii = 0; ii < int(n_rows); ++ii) {
            int const POS = ii - DEG;
            result += weights[ii]*hdcalib::Calib::evaluateSpline(val, POS, DEG);
        }
        return result;
    }

};

template<int NUM, int DEG>
void compareSplineVariants() {
    std::cout << "Comparing implementations for Spline-" << NUM << "-" << DEG << std::endl;
    runningstats::RunningStats errors;
    int const width = 100;
    int const height = 60;
    cv::Size const size(width, height);
    double maxval_x = 0;
    double maxval_y = 0;
    cv::Mat_<float> weights_x(NUM+DEG, NUM+DEG, 0.0);
    cv::Mat_<float> weights_y(NUM+DEG, NUM+DEG, 0.0);
    cv::randn(weights_x, 0, 1);
    cv::randn(weights_y, 0, 1);
    hdcalib::SplineFunctor<NUM, DEG> reference{cv::Vec2f(), cv::Vec2f(), size};
    SplineFunctor<NUM, DEG> new_impl{cv::Vec2f(), cv::Vec2f(), size};
    for (double tx = -1; tx <= width+1; tx += 0.5) {
        for (double ty = -1; ty <= height+1; ty += 0.5) {
            cv::Vec2f const new_im = new_impl.apply(cv::Vec2f(tx, ty), weights_x, weights_y);
            cv::Vec2f const ref = reference.apply(cv::Vec2f(tx, ty), weights_x, weights_y);
            ASSERT_NEAR(new_im[0], ref[0], 1e-6);
            ASSERT_NEAR(new_im[1], ref[1], 1e-6);
            errors.push_unsafe(cv::norm(new_im - ref));
            maxval_x = std::max<double>(maxval_x, ref[0]);
            maxval_y = std::max<double>(maxval_y, ref[1]);
        }
    }
    std::cout << "Errors: " << errors.print() << std::endl;
    std::cout << "Max val x, y: " << maxval_x << ", " << maxval_y << std::endl;
    ParallelTime t;
    for (double tx = -1; tx <= width+1; tx += 0.1) {
        for (double ty = -1; ty <= height+1; ty += 0.1) {
            cv::Vec2f const new_im = new_impl.apply(cv::Vec2f(tx, ty), weights_x, weights_y);
        }
    }
    std::cout << "New impl. time: " << t.print() << std::endl;
    t.start();
    for (double tx = -1; tx <= width+1; tx += 0.1) {
        for (double ty = -1; ty <= height+1; ty += 0.1) {
            cv::Vec2f const reference = new_impl.apply(cv::Vec2f(tx, ty), weights_x, weights_y);
        }
    }
    std::cout << "Reference impl. time: " << t.print() << std::endl;

    std::cout << std::endl;
}

TEST(Spline, normal_vs_reduced) {
    compareSplineVariants<3,3>();
    compareSplineVariants<5,3>();
    compareSplineVariants<7,3>();
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    std::cout << "RUN_ALL_TESTS return value: " << RUN_ALL_TESTS() << std::endl;
}
