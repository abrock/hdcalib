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

template<class T>
T evaluateSplineDerivative(T const x, int const POS, int const DEG) {
    if (x < T(POS) || x > T(POS+DEG+1) || 0 == DEG) {
        return T(0);
    }
    //*
    if (1 == DEG) {
        if (x < T(POS+1)) {
            return T(1);
        }
        return T(-1);
    }
    // */
    return evaluateSplineExplicit(x, POS, DEG-1) - evaluateSplineExplicit(x, POS+1, DEG-1);
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
        apply(result.val, weights_x.ptr<float>(), weights_y.ptr<float>(), size);
        return result;
    }

    static cv::Vec2f apply(cv::Vec2f const& pt,
                           cv::Mat_<float> const& weights_x,
                           cv::Mat_<float> weights_y,
                           cv::Size const& size) {
        cv::Vec2f result(pt);
        apply(result.val, weights_x.ptr<float>(), weights_y.ptr<float>(), size);
        return result;
    }

    cv::Vec2f apply(cv::Vec2f const& pt,
                    std::vector<double> const& weights_x,
                    std::vector<double> weights_y) const {
        cv::Vec2f result(pt);
        apply(result.val, weights_x.data(), weights_y.data(), size);
        return result;
    }

    static cv::Vec2f apply(cv::Vec2f const& pt,
                           std::vector<double> const& weights_x,
                           std::vector<double> weights_y,
                           cv::Size const& size) {
        cv::Vec2f result(pt);
        apply(result.val, weights_x.data(), weights_y.data(), size);
        return result;
    }

    template<class T>
    bool operator()(T const * const weights_x, T const * const weights_y, T * residuals) const {
        residuals[0] = T(src[0]);
        residuals[1] = T(src[1]);
        apply(residuals, weights_x, weights_y, size);
        residuals[0] -= T(dst[0]);
        residuals[1] -= T(dst[1]);
        return true;
    }

    template<class T, class U>
    static void apply(T* pt, U const * const weights_x, U const * const weights_y, cv::Size const& size) {
        T scaled_pt[2] = {pt[0]*T(double(NUM)/size.width), pt[1]*T(double(NUM)/size.height)};
        T const dx = applySingle(scaled_pt, weights_x);
        T const dy = applySingle(scaled_pt, weights_y);
        pt[0] += dx;
        pt[1] += dy;
    }

    template<class T, class U>
    static T apply_Dx(T const * const pt, U const * const weights, cv::Size const& size) {
        T scaled_pt[2] = {pt[0]*T(double(NUM)/size.width), pt[1]*T(double(NUM)/size.height)};
        return applySingleDx(scaled_pt, weights);
    }

    template<class T, class U>
    static T apply_Dy(T const * const pt, U const * const weights, cv::Size const& size) {
        T scaled_pt[2] = {pt[0]*T(double(NUM)/size.width), pt[1]*T(double(NUM)/size.height)};
        return applySingleDy(scaled_pt, weights);
    }

    template<class T, class U>
    static T applySingle(T const * const val, U const * const weights) {
        T col[SplineFunctor<NUM, DEG>::n_rows];
        for (size_t ii = 0; ii < SplineFunctor<NUM, DEG>::n_rows; ++ii) {
            col[ii] = applyRow(val[0], &weights[ii*SplineFunctor<NUM, DEG>::n_rows]);
        }
        return applyRow(val[1], col);
    }

    template<class T, class U>
    static T applySingleDx(T const * const val, U const * const weights) {
        T col[SplineFunctor<NUM, DEG>::n_rows];
        for (size_t ii = 0; ii < SplineFunctor<NUM, DEG>::n_rows; ++ii) {
            col[ii] = applyRowDerivative(val[0], &weights[ii*SplineFunctor<NUM, DEG>::n_rows]);
        }
        return applyRow(val[1], col);
    }

    template<class T, class U>
    static T applySingleDy(T const * const val, U const * const weights) {
        T col[SplineFunctor<NUM, DEG>::n_rows];
        for (size_t ii = 0; ii < SplineFunctor<NUM, DEG>::n_rows; ++ii) {
            col[ii] = applyRow(val[0], &weights[ii*SplineFunctor<NUM, DEG>::n_rows]);
        }
        return applyRowDerivative(val[1], col);
    }

    template<class T, class U>
    static T applyRow(T const& val, U const * const weights) {
        T result(0);
        for (int ii = 0; ii < int(SplineFunctor<NUM, DEG>::n_rows); ++ii) {
            int const POS = ii - DEG;
            result += weights[ii]*hdcalib::Calib::evaluateSpline(val, POS, DEG);
        }
        return result;
    }

    template<class T, class U>
    static T applyRowDerivative(T const& val, U const * const weights) {
        T result(0);
        for (int ii = 0; ii < int(SplineFunctor<NUM, DEG>::n_rows); ++ii) {
            int const POS = ii - DEG;
            result += weights[ii]*evaluateSplineDerivative(val, POS, DEG);
        }
        return result;
    }
};

template<int NUM, int DEG>
struct SplineRegularizer {
    cv::Size const size;
    SplineRegularizer(cv::Size const& _size) : size(_size) {}

    static int const n=6;

    template<class T>
    bool operator()(T const * const weights_x, T const * const weights_y, T * residuals) const {
        { // Calculate value at center
            T center_val[2] = {T(size.width)/T(2), T(size.height)/T(2)};
            SplineFunctor<NUM, DEG>::apply(center_val, weights_x, weights_y, size);
            residuals[0] = center_val[0] - T(size.width)/T(2);
            residuals[1] = center_val[1] - T(size.height)/T(2);
        }
        { //Add derivatives at center to residuals
            T center_val[2] = {T(size.width)/T(2), T(size.height)/T(2)};
            residuals[2] = SplineFunctor<NUM, DEG>::apply_Dx(center_val, weights_x, size);
            residuals[3] = SplineFunctor<NUM, DEG>::apply_Dy(center_val, weights_x, size);
            residuals[4] = SplineFunctor<NUM, DEG>::apply_Dx(center_val, weights_y, size);
            residuals[5] = SplineFunctor<NUM, DEG>::apply_Dy(center_val, weights_y, size);
        }
        for (int ii = 0; ii < n; ++ii) {
            residuals[ii] *= T(1000);
        }
        return true;
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
    for (double tx = -1; tx <= width+1; tx += 0.3) {
        for (double ty = -1; ty <= height+1; ty += 0.3) {
            cv::Vec2f const new_im = new_impl.apply(cv::Vec2f(tx, ty), weights_x, weights_y);
        }
    }
    std::cout << "New impl. time: " << t.print() << std::endl;
    t.start();
    for (double tx = -1; tx <= width+1; tx += 0.3) {
        for (double ty = -1; ty <= height+1; ty += 0.3) {
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

cv::Vec2f scaledDist(cv::Vec2f const& src, cv::Vec2f const& center, std::vector<double> const& dist) {
    cv::Vec2f result(0,0);
    double scale = cv::norm(center)*2;
    hdcalib::SemiFlexibleTargetProjectionFunctor::applyInverseDist((src/scale).val, result.val, (center/scale).val, dist.data());
    result *= scale;
    return result;
}

template<int NUM, int DEG>
void testFisheyeApprox() {
    std::cout << "Testing fisheye approx " << NUM << "-" << DEG << std::endl;
    std::vector<double> dist{3.6680970095817313e+00, -3.8722849263362034e+00,
                             -1.3864283696580085e-03, 7.0408416351859678e-04,
                             -9.4515171575199397e-01, 3.9625187906485353e+00,
                             -2.9129449112602157e+00, -2.2331052673279683e+00,
                             -1.2649868556908445e-02, 4.6521895829965314e-03,
                             1.2220152823402589e-02, -3.8804790887521337e-03,
                             -1.3619035602773649e-02, -1.5557989298651851e-02};

    for (double& d : dist) {
        d /= 1;
    }

    int const width = 160;
    int const height = 100;
    cv::Size size(width, height);
    cv::Vec2f const center(double(width-1)/2, double(height-1)/2);

    std::vector<cv::Vec2f> src_vec, dst_vec;

    typedef SplineFunctor<NUM, DEG> F;

    std::vector<double> weights_x(F::n, 0.0);
    std::vector<double> weights_y(F::n, 0.0);

    cv::Mat_<cv::Vec2f> flow(height, width, cv::Vec2f(0,0));

    ceres::Problem problem;
    for (int xx = 0; xx < width; xx += 3) {
        for (int yy = 0; yy < height; yy += 3) {
            cv::Vec2f src(xx, yy);
            cv::Vec2f dst = scaledDist(src, center, dist);
            src_vec.push_back(src);
            dst_vec.push_back(dst);
        }
    }
    for (size_t ii = 0; ii < src_vec.size(); ++ii) {
        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<F, 2, F::n, F::n>(
                                     new F(src_vec[ii], dst_vec[ii], size)),
                                 nullptr,
                                 weights_x.data(),
                                 weights_y.data()
                                 );
    }

    ceres::Solver::Options options;
    options.num_threads = int(8);
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.max_num_iterations = 500;
    options.minimizer_progress_to_stdout = true;
    double const ceres_tolerance = 1e-12;
    options.function_tolerance = ceres_tolerance;
    options.gradient_tolerance = ceres_tolerance;
    options.parameter_tolerance = ceres_tolerance;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << std::endl;
    std::cout << summary.FullReport() << std::endl;

    std::cout << "Plotting error img" << std::endl;

    runningstats::Image2D<double> error_img(1, 1);
    runningstats::RunningStats errors;
    runningstats::Image2D<double> gt_img(1, 1);
    runningstats::Image2D<double> fitted_img(1, 1);
    runningstats::QuantileStats<double> dist_movements;
    for (int xx = 0; xx < width; xx++) {
        for (int yy = 0; yy < height; yy++) {
            cv::Vec2f src(xx, yy);
            cv::Vec2f dst = scaledDist(src, center, dist);
            dist_movements.push_unsafe(cv::norm(dst - src));
            F f(src, dst, size);
            cv::Vec2d residual(0,0);
            f(weights_x.data(), weights_y.data(), residual.val);
            cv::Vec2f fitted = f.apply(src, weights_x, weights_y);
            double const error = cv::norm(residual);
            error_img[xx][yy] = error;
            errors.push_unsafe(error);
            gt_img[xx][yy] = cv::norm(dst - src);
            fitted_img[xx][yy] = cv::norm(fitted - src);
        }
    }
    std::cout << "Dist movements: " << dist_movements.print() << std::endl;
    std::cout << "Fit errors: " << errors.print() << std::endl;
    std::cout << "Saving error img" << std::endl;

    for (std::pair<std::string, cv::Vec2f> const& it : std::map<std::string, cv::Vec2f> {
    {"Center", center},
    {"left", cv::Vec2f(0, center[1])}
})
    {
        cv::Vec2f const value = F::apply(it.second, weights_x, weights_y, size);
        double const rot = F::apply_Dx(it.second.val, weights_y.data(), size) - F::apply_Dy(it.second.val, weights_x.data(), size);
        std::cout << "Value " << it.first << ": " << it.second << ", val: " << value << ", diff: " << it.second - value
                  << ", f(x,y)/dx: " << F::apply_Dx(it.second.val, weights_x.data(), size)
                  << ", " << F::apply_Dx(it.second.val, weights_y.data(), size)
                  << ", f(x,y)/dy: " << F::apply_Dy(it.second.val, weights_x.data(), size)
                  << ", " << F::apply_Dy(it.second.val, weights_y.data(), size)
                  << ", rot: " << rot
                  << std::endl;
    }

    error_img.plot(std::string("fit-errors-spline-") + std::to_string(NUM), runningstats::HistConfig());
    gt_img.plot(std::string("fit-errors-spline-") + std::to_string(NUM) + "-gt", runningstats::HistConfig());
    fitted_img.plot(std::string("fit-errors-spline-") + std::to_string(NUM) + "-fitted", runningstats::HistConfig());

    std::cout << "done" << std::endl;

}

template<int NUM, int DEG>
void testFisheyeApproxRegularized() {
    std::cout << "Testing fisheye approx " << NUM << "-" << DEG << std::endl;
    std::vector<double> dist{3.6680970095817313e+00, -3.8722849263362034e+00,
                             -1.3864283696580085e-03, 7.0408416351859678e-04,
                             -9.4515171575199397e-01, 3.9625187906485353e+00,
                             -2.9129449112602157e+00, -2.2331052673279683e+00,
                             -1.2649868556908445e-02, 4.6521895829965314e-03,
                             1.2220152823402589e-02, -3.8804790887521337e-03,
                             -1.3619035602773649e-02, -1.5557989298651851e-02};

    for (double& d : dist) {
        d /= 1;
    }

    int const width = 160;
    int const height = 100;
    cv::Size size(width, height);
    cv::Vec2f const center(double(width-1)/2, double(height-1)/2);

    std::vector<cv::Vec2f> src_vec, dst_vec;

    typedef SplineFunctor<NUM, DEG> F;
    typedef SplineRegularizer<NUM, DEG> Reg;

    std::vector<double> weights_x(F::n, 0.0);
    std::vector<double> weights_y(F::n, 0.0);

    cv::Mat_<cv::Vec2f> flow(height, width, cv::Vec2f(0,0));

    ceres::Problem problem;
    for (int xx = 0; xx < width; xx += 3) {
        for (int yy = 0; yy < height; yy += 3) {
            cv::Vec2f src(xx, yy);
            cv::Vec2f dst = scaledDist(src, center, dist);
            src_vec.push_back(src);
            dst_vec.push_back(dst);
        }
    }
    for (size_t ii = 0; ii < src_vec.size(); ++ii) {
        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<F, 2, F::n, F::n>(
                                     new F(src_vec[ii], dst_vec[ii], size)),
                                 nullptr,
                                 weights_x.data(),
                                 weights_y.data()
                                 );
    }

    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<Reg, Reg::n, F::n, F::n>(
                                 new Reg(size)),
                             nullptr,
                             weights_x.data(),
                             weights_y.data()
                             );

    ceres::Solver::Options options;
    options.num_threads = int(8);
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.max_num_iterations = 500;
    options.minimizer_progress_to_stdout = true;
    double const ceres_tolerance = 1e-12;
    options.function_tolerance = ceres_tolerance;
    options.gradient_tolerance = ceres_tolerance;
    options.parameter_tolerance = ceres_tolerance;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << std::endl;
    std::cout << summary.FullReport() << std::endl;

    std::cout << "Plotting error img" << std::endl;

    runningstats::Image2D<double> error_img(1, 1);
    runningstats::RunningStats errors;
    runningstats::Image2D<double> gt_img(1, 1);
    runningstats::Image2D<double> fitted_img(1, 1);
    runningstats::QuantileStats<double> dist_movements;
    for (int xx = 0; xx < width; xx++) {
        for (int yy = 0; yy < height; yy++) {
            cv::Vec2f src(xx, yy);
            cv::Vec2f dst = scaledDist(src, center, dist);
            dist_movements.push_unsafe(cv::norm(dst - src));
            F f(src, dst, size);
            cv::Vec2d residual(0,0);
            f(weights_x.data(), weights_y.data(), residual.val);
            cv::Vec2f fitted = f.apply(src, weights_x, weights_y);
            double const error = cv::norm(residual);
            error_img[xx][yy] = error;
            errors.push_unsafe(error);
            gt_img[xx][yy] = cv::norm(dst - src);
            fitted_img[xx][yy] = cv::norm(fitted - src);
        }
    }
    std::cout << "Dist movements: " << dist_movements.print() << std::endl;
    std::cout << "Fit errors: " << errors.print() << std::endl;
    std::cout << "Saving error img" << std::endl;

    for (std::pair<std::string, cv::Vec2f> const& it : std::map<std::string, cv::Vec2f> {
    {"Center", center},
    {"left", cv::Vec2f(0, center[1])}
})
    {
        cv::Vec2f const value = F::apply(it.second, weights_x, weights_y, size);
        double const rot = F::apply_Dx(it.second.val, weights_y.data(), size) - F::apply_Dy(it.second.val, weights_x.data(), size);
        std::cout << "Value " << it.first << ": " << it.second << ", val: " << value << ", diff: " << it.second - value
                  << ", f(x,y)/dx: " << F::apply_Dx(it.second.val, weights_x.data(), size)
                  << ", " << F::apply_Dx(it.second.val, weights_y.data(), size)
                  << ", f(x,y)/dy: " << F::apply_Dy(it.second.val, weights_x.data(), size)
                  << ", " << F::apply_Dy(it.second.val, weights_y.data(), size)
                  << ", rot: " << rot
                  << std::endl;
    }

    error_img.plot(std::string("fit-errors-spline-reg-") + std::to_string(NUM), runningstats::HistConfig());
    gt_img.plot(std::string("fit-errors-spline-reg-") + std::to_string(NUM) + "-gt", runningstats::HistConfig());
    fitted_img.plot(std::string("fit-errors-spline-reg-") + std::to_string(NUM) + "-fitted", runningstats::HistConfig());

    std::cout << "done" << std::endl;

}

TEST(Spline, fisheye_approximation) {
    testFisheyeApprox<3,3>();
    testFisheyeApprox<5,3>();
    testFisheyeApprox<7,3>();
    testFisheyeApprox<9,3>();
}

TEST(Spline, fisheye_approximation_regularized) {
    testFisheyeApproxRegularized<3,3>();
    testFisheyeApproxRegularized<5,3>();
    testFisheyeApproxRegularized<7,3>();
    testFisheyeApproxRegularized<9,3>();
}

TEST(Spline, derivative) {
    std::cout << "Testing spline derivatives" << std::endl;
    for (int DEG = 1; DEG <= 9; ++DEG) {
        runningstats::RunningStats errors;
        std::cout << "DEG: " << DEG << std::endl;
        for (int POS = -2; POS < DEG; ++POS) {
            for (double t = POS-2; t <= POS+DEG+3; t += .01) {
                ceres::Jet<double, 1> jet(t, 0);
                ceres::Jet<double, 1> result = evaluateSplineExplicit(jet, POS, DEG);
                double const deriv = hdcalib::Calib::evaluateSplineDerivative(t, POS, DEG);
                ASSERT_NEAR(result.v(0), deriv, 1e-6);
                errors.push_unsafe(std::abs(result.v(0) - deriv));
            }
        }
        std::cout << "errors: " << errors.print() << std::endl << std::endl;
    }
}

template<int NUM, int DEG>
void testDerivatives2D() {
    typedef SplineFunctor<NUM, DEG> F;
    std::cout << "Testing spline 2D derivatives" << std::endl;
    cv::Mat_<double> mat(F::n_rows, F::n_rows, 0.0);
    cv::randn(mat, 0, 1);

    runningstats::RunningCovariance covariance_dx, covariance_dy;
    runningstats::RunningStats errors_dx, errors_dy;
    for (double tx = -.5; tx <= double(NUM+DEG)+.5; tx += .1) {
        for (double ty = -.5; ty <= double(NUM+DEG)+.5; ty += .1) {
            ceres::Jet<double, 2> jet[2] = {{tx, 0}, {ty,1}};
            ceres::Jet<double, 2> result_jet = F::applySingle(jet, mat.ptr<double>());
            double xy[2] = {tx, ty};
            double const result_dx_explicit = F::applySingleDx(xy, mat.ptr<double>());
            double const result_dy_explicit = F::applySingleDy(xy, mat.ptr<double>());
            /*
            std::cout << tx << "\t" << ty << "\t"
                      << result_explicit << "\t" << result_jet.v[0] << "\t"
                      << std::abs(result_explicit - result_jet.v[0]) << std::endl;
            // */
            ASSERT_NEAR(result_dx_explicit, result_jet.v[0], 1e-6);
            covariance_dx.push_unsafe(result_dx_explicit, result_jet.v[0]);
            errors_dx.push_unsafe(std::abs(result_dx_explicit - result_jet.v[0]));

            ASSERT_NEAR(result_dy_explicit, result_jet.v[1], 1e-6);
            covariance_dy.push_unsafe(result_dy_explicit, result_jet.v[1]);
            errors_dy.push_unsafe(std::abs(result_dy_explicit - result_jet.v[1]));
        }
    }
    std::cout << "errors dx: " << errors_dx.print() << std::endl << std::endl;
    std::cout << "covariance dx: "; covariance_dx.printInfo(); std::cout << std::endl;

    std::cout << "errors dy: " << errors_dy.print() << std::endl << std::endl;
    std::cout << "covariance dy: "; covariance_dy.printInfo(); std::cout << std::endl;

}

TEST(Spline, derivative2D) {
    testDerivatives2D<3, 3>();
}

template<class T>
T poly(T const * const val) {
    T x = val[0];
    T y = val[1];
    return x*x + T(2)*y*y + T(3)*x*y + T(4)*x + T(5)*y + T(6);
}

template<class T>
T polyDy(T const * const val) {
    T x = val[0];
    T y = val[1];
    return T(4)*y + T(3)*x + T(5);
}

template<class T>
T polyDx(T const * const val) {
    T x = val[0];
    T y = val[1];
    return T(2)*x + T(3)*y + T(4);
}

TEST(Poly, derivative2D) {
    runningstats::RunningStats errors_dx, errors_dy;
    for (double x = -1; x <= 1; x += .01) {
        for (double y = -1; y <= 1; y += .01) {
            ceres::Jet<double, 2> val_jet[2];
            val_jet[0].a = x;
            val_jet[0].v[0] = 1;
            val_jet[0].v[1] = 0;

            val_jet[1].a = y;
            val_jet[1].v[0] = 0;
            val_jet[1].v[1] = 1;

            double val[2] = {x, y};
            ceres::Jet<double, 2> result = poly(val_jet);

            double const dx = polyDx(val);
            double const dy = polyDy(val);

            ASSERT_NEAR(dx, result.v[0], 1e-10);
            errors_dx.push_unsafe(dx - result.v[0]);

            ASSERT_NEAR(dy, result.v[1], 1e-10);
            errors_dy.push_unsafe(dy - result.v[1]);
        }
    }
    std::cout << "Errors dx: " << errors_dx.print() << std::endl;
    std::cout << "Errors dy: " << errors_dy.print() << std::endl;
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    std::cout << "RUN_ALL_TESTS return value: " << RUN_ALL_TESTS() << std::endl;
}
