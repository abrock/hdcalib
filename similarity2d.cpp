#include "hdcalib.h"

namespace hdcalib {

Point2d Similarity2D::transform(const Point2d src) {
    return Similarity2DCost::transform(angle, scale, t_x, t_y, src);
}

Similarity2D::Similarity2D(const std::vector<Point2d> &_src, const std::vector<Point2d> &_dst) : src(_src), dst(_dst) {
    if (src.size() != dst.size()) {
        throw std::runtime_error("Size of src (" + std::to_string(src.size()) + ") doesn't match size of dst (" + std::to_string(dst.size()) + ")");
    }
}

void Similarity2D::runFit() {
    ceres::Problem problem;

    if (src.size() != dst.size()) {
        throw std::runtime_error("Size of src (" + std::to_string(src.size()) + ") doesn't match size of dst (" + std::to_string(dst.size()) + ")");
    }

    for (size_t ii = 0; ii < src.size() && ii < dst.size(); ++ii) {
        ceres::CostFunction * cost_function =
                new ceres::AutoDiffCostFunction<
                Similarity2DCost,
                2,
                1, // focal length x
                1, // focal length y
                1, // principal point x
                1 // principal point y
                >(new Similarity2DCost(src[ii], dst[ii]));
        problem.AddResidualBlock(cost_function,
                                 cauchy_param > 0 ? new ceres::CauchyLoss(cauchy_param) : nullptr, // Loss function (nullptr = L2)
                                 &angle, // rotation angle
                                 &scale, // scale
                                 &t_x, // translation x
                                 &t_y // translation y
                                 );
    }

    ceres::Solver::Options options;
    options.num_threads = int(8);
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.max_num_iterations = 150;
    options.minimizer_progress_to_stdout = true;
    options.function_tolerance = ceres_tolerance;
    options.gradient_tolerance = ceres_tolerance;
    options.parameter_tolerance = ceres_tolerance;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    clog::L(__PRETTY_FUNCTION__, 1) << summary.BriefReport() << "\n";
    clog::L(__PRETTY_FUNCTION__, 1) << summary.FullReport() << "\n";
}

Similarity2D::Similarity2DCost::Similarity2DCost(const Point2d _src, const Point2d _dst) : src(_src), dst(_dst) {}

Point2d Similarity2D::Similarity2DCost::transform(const double angle, const double scale, const double t_x, const double t_y, const Point2d src) {
    cv::Point2d result(0,0);
    transform(angle, scale, t_x, t_y, src, result.x, result.y);
    return result;
}

template<class T>
bool Similarity2D::Similarity2DCost::operator ()(
        T const * const angle,
        T const * const scale,
        T const * const t_x,
        T const * const t_y,
        T *residuals
        ) const {
    transform(angle[0], scale[0], t_x[0], t_y[0], src, residuals[0], residuals[1]);
    residuals[0] -= dst.x;
    residuals[1] -= dst.y;
    return true;
}

template<class T>
void Similarity2D::Similarity2DCost::transform(const T &angle, const T &scale, const T &t_x, const T &t_y, const Point2d &src, T &dst_x, T &dst_y) {
    T const sin = ceres::sin(angle);
    T const cos = ceres::cos(angle);
    dst_x = scale * (cos * src.x - sin*src.y) + t_x;
    dst_y = scale * (sin * src.x + cos*src.y) + t_y;
}

} // namespace hdcalib
