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

    ignored = 0;

    for (size_t ii = 0; ii < src.size() && ii < dst.size(); ++ii) {
        if (outlier_threshold > 0) {
            cv::Point2d const initial_res = transform(src[ii]) - dst[ii];
            double const initial_dist = std::sqrt(initial_res.dot(initial_res));
            if (initial_dist > outlier_threshold) {
                ignored++;
                continue;
            }
        }
        ceres::CostFunction * cost_function =
                new ceres::AutoDiffCostFunction<
                Similarity2DCost,
                2,
                1, // angle
                1, // scale
                1, // translate x
                1 // translate y
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
    options.minimizer_progress_to_stdout = verbose;
    options.function_tolerance = ceres_tolerance;
    options.gradient_tolerance = ceres_tolerance;
    options.parameter_tolerance = ceres_tolerance;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    t_length = std::sqrt(t_x * t_x + t_y*t_y);

    if (verbose) {
        clog::L(__PRETTY_FUNCTION__, 1) << summary.BriefReport() << "\n";
        clog::L(__PRETTY_FUNCTION__, 1) << summary.FullReport() << "\n";
    }

    max_movement = 0;
    max_scale_movement = 0;
    max_rotate_movement = 0;
    Similarity2D scaler, rotator;
    scaler.scale = scale;
    rotator.angle = angle;

    for (cv::Point2d const& s : src) {
        {
            cv::Point2d const d = transform(s);
            cv::Point2d const diff = d - s;
            double const length = std::sqrt(diff.dot(diff));
            if (length > max_movement) {
                max_movement = length;
                max_movement_pt = diff;
            }
        }
        {
            cv::Point2d const d = scaler.transform(s);
            cv::Point2d const diff = d - s;
            double const length = std::sqrt(diff.dot(diff));
            if (length > max_scale_movement) {
                max_scale_movement = length;
            }
        }
        {
            cv::Point2d const d = rotator.transform(s);
            cv::Point2d const diff = d - s;
            double const length = std::sqrt(diff.dot(diff));
            if (length > max_rotate_movement) {
                max_rotate_movement = length;
            }
        }
    }
}

void Similarity2D::print(std::ostream& out) const {
    out << "translate: (" << t_x << ", " << t_y << ") / " << t_length
        << ", angle: " << angle*180.0/M_PI << "°, scale: " << scale
        << " max movement: " << max_movement_pt << " / " << max_movement
        << ", max scale move: " << max_scale_movement
        << ", max rotate move: " << max_rotate_movement
        << ", ignored: " << 100.0*double(ignored) / src.size() << "%" << std::endl;
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
