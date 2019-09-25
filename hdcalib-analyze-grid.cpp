#include "hdcalib.h"

namespace hdcalib {


struct GridCost {
    int const row;
    int const col;

    cv::Vec3d const point;
    cv::Vec3d const center;

    double const weight;


    GridCost(
            int const _row,
            int const _col,
            cv::Vec3d const & _point,
            cv::Vec3d const & _center,
            double const _weight) :
        row(_row),
        col(_col),
        point(_point),
        center(_center),
        weight(_weight) {}

    template<class T>
    bool operator () (
            T const * const row_vec,
            T const * const col_vec,
            T * residuals) const {
        for (size_t ii = 0; ii < 3; ++ii) {
            residuals[ii] = weight * (T(point[ii]) - (T(center[ii]) + T(row) * row_vec[ii] + T(col) * col_vec[ii]));
        }
        return true;
    }
};


/**
 * @brief The GridCostFreeCenter struct Does not work, don't use it. Estimation of the LF grid with the center not fixed.
 */
struct GridCostFreeCenter {
    int const row;
    int const col;

    cv::Vec3d const p;
    cv::Vec3d const p0;

    GridCostFreeCenter(int const _row, int const _col, cv::Vec3d const & _p, cv::Vec3d const & _p0) : row(_row), col(_col), p(_p), p0(_p0) {}

    template<class T>
    bool operator () (
            T const * const row_vec,
            T const * const col_vec,
            T const * const center,
            T * residuals) const {
        for (size_t ii = 0; ii < 3; ++ii) {
            residuals[ii] = T(p[ii]) - (T(p0[ii]) + T(row) * row_vec[ii] + T(col) * col_vec[ii]);
        }
        return true;
    }
};

void Calib::analyzeGridLF(const size_t rows, const size_t cols, const std::vector<string> &images) {
    cv::Vec3d row_vec, col_vec;
    getGridVectors(rows, cols, images, row_vec, col_vec);

    cv::Vec3d row_vec2, col_vec2;
    getGridVectors2(rows, cols, images, row_vec2, col_vec2);

    cv::Vec3d rect_rot;
    getRectificationRotation(rows, cols, images, rect_rot);


    getIndividualRectificationRotation(rows, cols, images, rect_rot);

    { // getGridVectors()
        double const row_vec_length = std::sqrt(row_vec.dot(row_vec));
        double const col_vec_length = std::sqrt(col_vec.dot(col_vec));
        clog::L(__func__, 1) << "getGridVectors():" << std::endl;
        clog::L(__func__, 1) << "row_vec: " << row_vec << ", length: " << row_vec_length << std::endl
                                    << "col_vec: " << col_vec << ", length: " << col_vec_length << std::endl;

        double cos_alpha = row_vec.dot(col_vec) / (row_vec_length * col_vec_length);
        clog::L(__func__, 1) << "cos alpha: " << cos_alpha << std::endl
                                    << "angle: " << std::acos(cos_alpha)/M_PI*180. << std::endl;
    }
    { // getGridVectors2()
        double const row_vec_length = std::sqrt(row_vec2.dot(row_vec2));
        double const col_vec_length = std::sqrt(col_vec2.dot(col_vec2));
        clog::L(__func__, 1) << "getGridVectors2():" << std::endl
                                    << "row_vec: " << row_vec2 << ", length: " << row_vec_length << std::endl
                                    << "col_vec: " << col_vec2 << ", length: " << col_vec_length << std::endl;

        double cos_alpha = row_vec2.dot(col_vec2) / (row_vec_length * col_vec_length);
        clog::L(__func__, 1) << "cos alpha: " << cos_alpha << std::endl
                                    << "angle: " << std::acos(cos_alpha)/M_PI*180. << std::endl;
    }
}

void Calib::getGridVectors(const size_t rows, const size_t cols, const std::vector<string> &images, Vec3d &row_vec, Vec3d &col_vec) {
    prepareCalibration();
    ceres::Problem problem;

    std::string const& middle_name = images[images.size()/2];
    CornerStore const& middle = data[middle_name];
    size_t const& middle_id = getId(middle_name);

    std::map<std::string, std::vector<GridCost* > > cost_functions;

    size_t counter = 0;
    for (int row = -int(rows/2); row <= int(rows/2); ++row) {
        for (int col = -int(cols/2); col <= int(cols/2); ++col, ++counter) {
            clog::L(__func__, 2) << "row: " << row << ", col: " << col << ", counter: " << counter << ", name: " << images[counter] << std::endl;
            if (0 == col && 0 == row) {
                continue;
            }
            CornerStore const& current = data[images[counter]];
            size_t const id = getId(images[counter]);
            std::vector<GridCost* > & target_costs = cost_functions[images[counter]];
            std::vector<cv::Vec3d> src_vec, dst_vec;
            for (size_t ii = 0; ii < middle.size(); ++ii) {
                hdmarker::Corner const& src = middle.get(ii);
                std::vector<hdmarker::Corner> const _dst = current.findByID(src);
                if (_dst.empty()) {
                    continue;
                }
                //const F p[], T result[], const T R[], const T t[];
                hdmarker::Corner const& dst = _dst.front();
                if (src.page != dst.page || src.id.x != dst.id.x || src.id.y != dst.id.y) {
                    continue;
                }
                src_vec.push_back(get3DPoint(src, rvecs[middle_id], tvecs[middle_id]));
                dst_vec.push_back(get3DPoint(dst, rvecs[id], tvecs[id]));
            }
            for (size_t ii = 0; ii < src_vec.size(); ++ii) {
                GridCost * cost = new GridCost(row, col, src_vec[ii], dst_vec[ii], 1.0/src_vec.size());
                ceres::CostFunction * cost_function = new ceres::AutoDiffCostFunction<GridCost, 3, 3, 3>(
                            cost
                            );
                problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(.001), row_vec.val, col_vec.val);
                target_costs.push_back(cost);
            }
        }
    }
    ceres::Solver::Options options;
    options.num_threads = threads;
    options.max_num_iterations = 150;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    options.function_tolerance = ceres_tolerance;
    options.gradient_tolerance = ceres_tolerance;
    options.parameter_tolerance = ceres_tolerance;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    clog::L(__func__, 1) << summary.FullReport() << "\n";

    /* Global residual statistics */
    runningstats::QuantileStats<double> g_res[3];
    runningstats::QuantileStats<double> g_err[3];
    for (const auto& it : cost_functions) {
        /* Local residual statistics */
        runningstats::QuantileStats<double> res[3];
        runningstats::QuantileStats<double> err[3];

        for (const auto cost : it.second) {
            double residuals[3];
            cost->operator()(row_vec.val, col_vec.val, residuals);
            for (size_t ii = 0; ii < 3; ++ii) {
                g_res[ii].push_unsafe(residuals[ii]*1000);
                res[ii].push_unsafe(residuals[ii]*1000);
                g_err[ii].push_unsafe(std::abs(residuals[ii])*1000);
                err[ii].push_unsafe(std::abs(residuals[ii])*1000);
            }
        }

        clog::L(__func__, 2) << "Residual stats for " << it.first << ":" << std::endl;
        for (size_t ii = 0; ii < 3; ++ii) {
            clog::L(__func__, 2) << res[ii].print() << std::endl;
        }
        clog::L(__func__, 2) << "Error stats for " << it.first << ":" << std::endl;
        for (size_t ii = 0; ii < 3; ++ii) {
            clog::L(__func__, 2) << err[ii].print() << std::endl;
        }
        clog::L(__func__, 2) << std::endl;
    }
    clog::L(__func__, 1) << "Residual stats for all images:" << std::endl;
    for (size_t ii = 0; ii < 3; ++ii) {
        clog::L(__func__, 1) << g_res[ii].print() << std::endl;
    }
    clog::L(__func__, 1) << "Error stats for all images:" << std::endl;
    for (size_t ii = 0; ii < 3; ++ii) {
        clog::L(__func__, 1) << g_err[ii].print() << std::endl;
    }
    clog::L(__func__, 1) << std::endl;

    double const row_vec_length = std::sqrt(row_vec.dot(row_vec));
    double const col_vec_length = std::sqrt(col_vec.dot(col_vec));
    clog::L(__func__, 1) << "row_vec: " << row_vec << ", length: " << row_vec_length << std::endl
                                 << "col_vec: " << col_vec << ", length: " << col_vec_length << std::endl;

    double cos_alpha = row_vec.dot(col_vec) / (row_vec_length * col_vec_length);
    clog::L(__func__, 1) << "cos alpha: " << cos_alpha << std::endl
                                 << "angle: " << std::acos(cos_alpha)/M_PI*180. << std::endl;
}

void Calib::getGridVectors2(const size_t rows, const size_t cols, const std::vector<string> &images, Vec3d &row_vec, Vec3d &col_vec) {
    prepareCalibration();
    ceres::Problem problem;

    std::string const& middle_name = images[images.size()/2];
    CornerStore const& middle = data[middle_name];
    size_t const& middle_id = getId(middle_name);

    std::map<std::string, std::vector<GridCost* > > cost_functions;

    std::vector<runningstats::BinaryStats> middle_marker_percentage(images.size());
    std::vector<hdmarker::Corner> intersection;
    for (hdmarker::Corner const& c : middle.getCorners()) {
        bool is_in_intersection = true;
        for (size_t ii = 0; ii < images.size(); ++ii) {
            if (!data[images[ii]].hasID(c)) {
                is_in_intersection = false;
                middle_marker_percentage[ii].push(false);
            }
            else {
                middle_marker_percentage[ii].push(true);
            }
        }
        if (is_in_intersection) {
            intersection.push_back(c);
        }
    }

    size_t counter = 0;
    for (int row = -int(rows/2); row <= int(rows/2); ++row) {
        for (int col = -int(cols/2); col <= int(cols/2); ++col, ++counter) {
            clog::L(__func__, 2) << "row: " << row << ", col: " << col << ", counter: " << counter << ", name: " << images[counter] << std::endl;
            if (0 == col && 0 == row) {
                continue;
            }
            CornerStore const& current = data[images[counter]];
            size_t const id = getId(images[counter]);
            std::vector<GridCost* > & target_costs = cost_functions[images[counter]];
            std::vector<cv::Vec3d> src_vec, dst_vec;
            for (size_t ii = 0; ii < middle.size(); ++ii) {
                hdmarker::Corner const& src = middle.get(ii);
                std::vector<hdmarker::Corner> const _dst = current.findByID(src);
                if (_dst.empty()) {
                    continue;
                }
                //const F p[], T result[], const T R[], const T t[];
                hdmarker::Corner const& dst = _dst.front();
                if (src.page != dst.page || src.id.x != dst.id.x || src.id.y != dst.id.y) {
                    continue;
                }
                src_vec.push_back(get3DPoint(src, rvecs[middle_id], tvecs[middle_id]));
                dst_vec.push_back(get3DPoint(dst, rvecs[id], tvecs[id]));
            }
            for (size_t ii = 0; ii < src_vec.size(); ++ii) {
                GridCost * cost = new GridCost(row, col, src_vec[ii], dst_vec[ii], 1.0/src_vec.size());
                ceres::CostFunction * cost_function = new ceres::AutoDiffCostFunction<GridCost, 3, 3, 3>(
                            cost
                            );
                problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(.001), row_vec.val, col_vec.val);
                target_costs.push_back(cost);
            }
        }
    }
    ceres::Solver::Options options;
    options.num_threads = threads;
    options.max_num_iterations = 150;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    options.function_tolerance = ceres_tolerance;
    options.gradient_tolerance = ceres_tolerance;
    options.parameter_tolerance = ceres_tolerance;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    clog::L(__func__, 1) << summary.FullReport() << "\n";

    /* Global residual statistics */
    runningstats::QuantileStats<double> g_res[3];
    runningstats::QuantileStats<double> g_err[3];
    for (const auto& it : cost_functions) {
        /* Local residual statistics */
        runningstats::QuantileStats<double> res[3];
        runningstats::QuantileStats<double> err[3];

        for (const auto cost : it.second) {
            double residuals[3];
            cost->operator()(row_vec.val, col_vec.val, residuals);
            for (size_t ii = 0; ii < 3; ++ii) {
                g_res[ii].push_unsafe(residuals[ii]*1000);
                res[ii].push_unsafe(residuals[ii]*1000);
                g_err[ii].push_unsafe(std::abs(residuals[ii])*1000);
                err[ii].push_unsafe(std::abs(residuals[ii])*1000);
            }
        }

        clog::L(__func__, 2) << "Residual stats for " << it.first << ":" << std::endl;
        for (size_t ii = 0; ii < 3; ++ii) {
            clog::L(__func__, 2) << res[ii].print() << std::endl;
        }
        clog::L(__func__, 2) << "Error stats for " << it.first << ":" << std::endl;
        for (size_t ii = 0; ii < 3; ++ii) {
            clog::L(__func__, 2) << err[ii].print() << std::endl;
        }
        clog::L(__func__, 2) << std::endl;
    }
    clog::L(__func__, 1) << "Residual stats for all images:" << std::endl;
    for (size_t ii = 0; ii < 3; ++ii) {
        clog::L(__func__, 1) << g_res[ii].print() << std::endl;
    }
    clog::L(__func__, 1) << "Error stats for all images:" << std::endl;
    for (size_t ii = 0; ii < 3; ++ii) {
        clog::L(__func__, 1) << g_err[ii].print() << std::endl;
    }
    clog::L(__func__, 1) << std::endl;

    clog::L(__func__, 1) << "Intersection size: " << intersection.size() << std::endl;

    for (size_t ii = 0; ii < images.size(); ++ii) {
        clog::L(__func__, 1) << "Corners in image " << images[ii] << ": " << data[images[ii]].size() << ", shares " << middle_marker_percentage[ii].getPercent() << "% with center view" <<  std::endl;
    }

    double const row_vec_length = std::sqrt(row_vec.dot(row_vec));
    double const col_vec_length = std::sqrt(col_vec.dot(col_vec));
    clog::L(__func__, 1) << "row_vec: " << row_vec << ", length: " << row_vec_length << std::endl
            << "col_vec: " << col_vec << ", length: " << col_vec_length << std::endl;

    double cos_alpha = row_vec.dot(col_vec) / (row_vec_length * col_vec_length);
    clog::L(__func__, 1) << "cos alpha: " << cos_alpha << std::endl;
    clog::L(__func__, 1) << "angle: " << std::acos(cos_alpha)/M_PI*180. << std::endl;
}

} // namespace hdcalib
