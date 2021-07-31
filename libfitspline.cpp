#include "libplotoptflow.h"

#include <ceres/ceres.h>

#include "gnuplot-iostream.h"

namespace hdflow {

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
            col[ii] = applyRow(val[0], &weights[ii*n_rows]);
        }
        return applyRow(val[1], col);
    }

    template<class T, class U>
    T applyRow(T const& val, U const * const weights) const {
        T result(0);
        for (int ii = 0; ii < n_rows; ++ii) {
            int POS = ii - DEG;
            result += weights[ii]*evaluateSplineExplicit(val, POS, DEG);
        }
        return result;
    }

};

template<int NUM, int DEG>
void fitSpline(
        std::string const prefix,
        const cv::Mat_<cv::Vec2f> &flow,
        double factor,
        const double length_factor,
        const cv::Scalar &color) {

    int const n_rows = SplineFunctor<NUM,DEG>::n_rows;
    std::vector<double> weights_x(n_rows*n_rows, 0.0);
    std::vector<double> weights_y(n_rows*n_rows, 0.0);

    runningstats::QuantileStats<float> motion_stats, abs_motion_stats, length_stats;
    for (cv::Vec2f const& it : flow) {
        if (std::isfinite(it[0]) && std::isfinite(it[1])) {
            motion_stats.push_unsafe(it[0]);
            motion_stats.push_unsafe(it[1]);
            abs_motion_stats.push_unsafe(std::abs(it[0]));
            abs_motion_stats.push_unsafe(std::abs(it[1]));
            length_stats.push_unsafe(std::sqrt(it[0]*it[0] + it[1]*it[1]));
        }
    }
    if (factor <= 0) {
        factor = length_stats.getQuantile(.999);
    }
    std::cout << "Factor: " << factor << std::endl;
    cv::Mat_<cv::Vec3b> result = colorFlow(flow, factor, 1);

    ceres::Problem problem;

    std::string const yaml_cache = prefix + "-spline-" + std::to_string(NUM) + "-" + std::to_string(DEG) + ".yaml";
    try {
        cv::FileStorage input(yaml_cache, cv::FileStorage::READ);
        input["weights_x"] >> weights_x;
        input["weights_y"] >> weights_y;
    }
    catch (...) {

    }

    cv::Vec2f const center{float(result.cols-1)/2, float(result.rows-1)/2};
    for (int row = 0; row < result.rows; row++) {
        for (int col = 0; col < result.cols; col++) {
            cv::Vec2f dst{float(col), float(row)};
            cv::Vec2f src = dst + flow(row, col);
            problem.AddResidualBlock(
                        new ceres::AutoDiffCostFunction<SplineFunctor<NUM,DEG>, 2, SplineFunctor<NUM,DEG>::n, SplineFunctor<NUM,DEG>::n>(
                            new SplineFunctor<NUM,DEG>(src, dst, result.size())
                            ),
                        nullptr,
                        weights_x.data(),
                        weights_y.data()
                        );
        }
    }


    // Run the solver!
    ceres::Solver::Options options;
    options.num_threads = int(8);
    options.max_num_iterations = 500;
    double const ceres_tolerance = 1e-12;
    options.function_tolerance = ceres_tolerance;
    options.gradient_tolerance = ceres_tolerance;
    options.parameter_tolerance = ceres_tolerance;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    cv::FileStorage output(yaml_cache, cv::FileStorage::WRITE);
    output << "weights_x" << weights_x;
    output << "weights_y" << weights_y;

    std::cout << summary.BriefReport() << "\n";
    std::cout << summary.FullReport() << "\n";

    cv::Mat_<cv::Vec2f> corrected(flow.size());
    runningstats::QuantileStats<float> corrected_stats;
    for (int row = 0; row < result.rows; row++) {
        for (int col = 0; col < result.cols; col++) {
            if (!std::isfinite(flow(row,col)[0]) || !std::isfinite(flow(row,col)[1])) {
                continue;
            }
            cv::Vec2f dst{float(col), float(row)};
            cv::Vec2f src = dst + flow(row, col);
            SplineFunctor<NUM,DEG> func(src, dst, result.size());
            corrected(row, col) = func.apply(src, weights_x, weights_y) - dst;
            corrected_stats.push_unsafe(cv::norm(corrected(row,col)));
        }
    }

    //cv::imwrite(prefix + "-orig.png", plotWithArrows(flow, factor, length_factor, color));
    cv::imwrite(prefix + "-corrected-spline.png", plotWithArrows(corrected, corrected_stats.getQuantile(.9), length_factor, color));

    //gnuplotWithArrows(prefix + "-orig-gpl", flow, factor, length_factor);
    gnuplotWithArrows(prefix + "-corrected-spline-gpl", corrected, corrected_stats.getQuantile(.9), length_factor);
}

template void fitSpline<9,3>(
        std::string const prefix,
        const cv::Mat_<cv::Vec2f> &flow,
        double factor,
        const double length_factor,
        const cv::Scalar &color);

} // namespace hdflow
