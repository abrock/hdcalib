#include "hdcalib.h"

namespace hdcalib {


Calib::RectifyCost::RectifyCost(const int8_t _axis, const Vec3d _src_a, const Vec3d _src_b, const cv::Mat_<double> &_cameraMatrix, const float _weight) :
    axis(_axis),
    src_a(_src_a),
    src_b(_src_b),
    cameraMatrix(_cameraMatrix),
    weight(_weight) {}

template<class T>
void Calib::RectifyCost::compute(
        T const * const rot_vec,
        T * result1,
        T * result2
        ) const {
    T focal[2] = {T(cameraMatrix(0,0)), T(cameraMatrix(1,1))};
    T principal[2] = {T(cameraMatrix(0,2)), T(cameraMatrix(1,2))};

    T p1[3] = {T(src_a[0]), T(src_a[1]), T(src_a[2])};
    T p2[3] = {T(src_b[0]), T(src_b[1]), T(src_b[2])};

    T rot_mat[9];
    Calib::rot_vec2mat(rot_vec, rot_mat);

    T translation[3] = {T(0), T(0), T(0)};

    Calib::project(p1, result1, focal, principal, rot_mat, translation);
    Calib::project(p2, result2, focal, principal, rot_mat, translation);
}

template<class T>
bool Calib::RectifyCost::operator () (
        T const * const rot_vec,
        T * residuals) const {


    T result1[2] = {T(0), T(0)};
    T result2[2] = {T(0), T(0)};

    compute(rot_vec, result1, result2);

    residuals[0] = T(weight)*(result1[1-axis] - result2[1-axis]);

    return true;
}

template<class RCOST>
void Calib::addImagePairToRectificationProblem(
        CalibResult & calib,
        CornerStore const& current,
        size_t const current_id,
        CornerStore const& next,
        size_t const next_id,
        std::vector<RCOST* > & target_costs,
        ceres::Problem & problem,
        int8_t const axis,
        double rot_vec[3]
) {
    size_t intersection_counter = 0;
    for (size_t ii = 0; ii < current.size(); ++ii) {
        hdmarker::Corner const& src = current.get(ii);
        std::vector<hdmarker::Corner> const _dst = next.findByID(src);
        if (_dst.empty()) {
            continue;
        }
        //const F p[], T result[], const T R[], const T t[];
        hdmarker::Corner const& dst = _dst.front();
        if (src.page != dst.page || src.id.x != dst.id.x || src.id.y != dst.id.y) {
            continue;
        }
        intersection_counter++;
    }

    float const weight = 1.0f/intersection_counter;
    for (size_t ii = 0; ii < current.size(); ++ii) {
        hdmarker::Corner const& src = current.get(ii);
        std::vector<hdmarker::Corner> const _dst = next.findByID(src);
        if (_dst.empty()) {
            continue;
        }
        //const F p[], T result[], const T R[], const T t[];
        hdmarker::Corner const& dst = _dst.front();
        if (src.page != dst.page || src.id.x != dst.id.x || src.id.y != dst.id.y) {
            continue;
        }

        cv::Vec3d src1, dst1;
        src1 = get3DPoint(calib, src, calib.rvecs[current_id], calib.tvecs[current_id]);
        dst1 = get3DPoint(calib, dst, calib.rvecs[next_id], calib.tvecs[next_id]);
        RectifyCost * cost = new RectifyCost(axis, src1, dst1, calib.cameraMatrix, weight);
        ceres::CostFunction * cost_function = new ceres::AutoDiffCostFunction<RectifyCost, 1, 3>(
                    cost
                    );
        //ceres::LossFunction * loss_func = new ceres::CauchyLoss(10);
        ceres::LossFunction * loss_func = nullptr;

        problem.AddResidualBlock(cost_function, loss_func, rot_vec);
        target_costs.push_back(cost);
    }
}

void Calib::rectificationResidualsPlotsAndStats(
        const char * log_name,
        std::map<std::string, std::vector<RectifyCost* > > const& cost_functions,
        double rot_vec[3],
bool plot
) {
    /* Global residual statistics */
    runningstats::QuantileStats<double> g_res[2];
    runningstats::QuantileStats<double> g_err[2];
    size_t counter = 0;
    for (const auto& it : cost_functions) {
        /* Local residual statistics */
        runningstats::QuantileStats<double> res[2];
        runningstats::QuantileStats<double> err[2];

        for (const auto cost : it.second) {
            double result1[2] = {0,0};
            double result2[2] = {0,0};
            cost->compute(rot_vec, result1, result2);

            double const residual = result1[1 - cost->axis] - result2[1 - cost->axis];

            g_res[cost->axis].push_unsafe(residual);
            res[cost->axis].push_unsafe(residual);

            g_err[cost->axis].push_unsafe(std::abs(residual));
            err[cost->axis].push_unsafe(std::abs(residual));
        }

        clog::L(log_name, 2) << "Residual stats for " << it.first << ":" << std::endl;
        for (size_t ii = 0; ii < 2; ++ii) {
            clog::L(log_name, 2) << res[ii].print() << std::endl;
        }
        clog::L(log_name, 2) << "Error stats for " << it.first << ":" << std::endl;
        for (size_t ii = 0; ii < 2; ++ii) {
            clog::L(log_name, 2) << err[ii].print() << std::endl;
        }
        for (size_t ii = 0; ii < 2 && plot; ++ii) {
            res[ii].plotHistAndCDF(std::string("rect-local-residuals-")
                                   + std::to_string(counter) + "-" + std::to_string(ii), .1);
        }
        counter++;
    }
    clog::L(log_name, 1) << "Residual stats for all images:" << std::endl;
    for (size_t ii = 0; ii < 2; ++ii) {
        clog::L(log_name, 1) << g_res[ii].print() << std::endl;
        if (plot) {
            g_res[ii].plotHistAndCDF(std::string("rect-global-residuals-") + std::to_string(ii), .1);
            g_err[ii].plotHistAndCDF(std::string("rect-global-errors-") + std::to_string(ii), .1);
        }
    }
    clog::L(log_name, 1) << "Error stats for all images:" << std::endl;
    for (size_t ii = 0; ii < 2; ++ii) {
        clog::L(log_name, 1) << g_err[ii].print() << std::endl;
    }
    clog::L(log_name, 1) << std::endl;
}

void Calib::getRectificationRotation(
        CalibResult & calib,
        const size_t rows,
        const size_t cols,
        const std::vector<std::string> &images,
        cv::Vec3d &rect_rot) {
    prepareCalibration();
    ceres::Problem problem;

    std::map<std::string, std::vector<RectifyCost* > > cost_functions;

    double rot_vec[3] = {0.001, 0.001, -0.001};

    size_t counter = 0;
    for (int row = -int(rows/2); row <= int(rows/2); ++row) {
        for (int col = -int(cols/2); col <= int(cols/2); ++col, ++counter) {
            clog::L(__func__, 2) << "row: " << row << ", col: " << col << ", counter: " << counter << ", name: " << images[counter] << std::endl;
            if (col == int(cols/2)) {
                continue;
            }
            CornerStore const& current = data[images[counter]];
            size_t const current_id = getId(images[counter]);
            CornerStore const& next = data[images[counter+1]];
            size_t const next_id = getId(images[counter+1]);
            std::vector<RectifyCost* > & target_costs = cost_functions[images[counter]];

            addImagePairToRectificationProblem(
                        calib,
                        current,
                        current_id,
                        next,
                        next_id,
                        target_costs,
                        problem,
                        0,
                        rot_vec);
        }
    }
    counter = 0;
    for (int row = -int(rows/2); row <= int(rows/2); ++row) {
        for (int col = -int(cols/2); col <= int(cols/2); ++col, ++counter) {
            clog::L(__func__, 2) << "row: " << row << ", col: " << col << ", counter: " << counter << ", name: " << images[counter] << std::endl;
            if (row == int(rows/2)) {
                continue;
            }
            CornerStore const& current = data[images[counter]];
            size_t const current_id = getId(images[counter]);
            CornerStore const& next = data[images[counter+cols]];
            size_t const next_id = getId(images[counter+cols]);
            std::vector<RectifyCost* > & target_costs = cost_functions[images[counter]];

            addImagePairToRectificationProblem(
                        calib,
                        current,
                        current_id,
                        next,
                        next_id,
                        target_costs,
                        problem,
                        1,
                        rot_vec);
        }
    }

    clog::L(__func__, 1) << "Error and residual stats before optimization" << std::flush;
    rectificationResidualsPlotsAndStats(
                __func__,
                cost_functions,
                rot_vec,
                false
                );


    ceres::Solver::Options options;
    options.num_threads = int(threads);
    options.max_num_iterations = 150;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    options.function_tolerance = ceres_tolerance;
    options.gradient_tolerance = ceres_tolerance;
    options.parameter_tolerance = ceres_tolerance;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    clog::L(__func__, 1) << summary.FullReport() << "\n";

    Calib::normalizeRotationVector(rot_vec);

    std::ofstream global_out_1("rect-values1.data");
    global_out_1 << "#Residuals for the x-axis 1-2. x/y value of [estimated 3D-position of marker projected to image plance]. 3-4. same for second value. 5.: residual" << std::endl;
    std::ofstream global_out_2("rect-values2.data");
    global_out_2 << "#Residuals for the y-axis 1-2. x/y value of [estimated 3D-position of marker projected to image plance]. 3-4. same for second value. 5.: residual" << std::endl;

    rectificationResidualsPlotsAndStats(
                __func__,
                cost_functions,
                rot_vec,
                true
                );


    Calib::normalizeRotationVector(rot_vec);

    rect_rot = cv::Vec3d(rot_vec[0], rot_vec[1], rot_vec[2]);
    calib.rectification = cv::Mat_<double>{rot_vec[0], rot_vec[1], rot_vec[2]};

double const degree = std::sqrt(rect_rot.dot(rect_rot)) / M_PI * 180;


clog::L(__func__, 1) << "Rotation vector: " << rot_vec[0] << ", " << rot_vec[1] << ", " << rot_vec[2] << std::endl;
clog::L(__func__, 1) << "Rotation: " <<  degree << "°" << std::endl;
}

void Calib::getIndividualRectificationRotation(
        CalibResult & calib,
        const size_t rows,
        const size_t cols,
        const std::vector<std::string> &images,
        cv::Vec3d &rect_rot) {
    prepareCalibration();
    ceres::Problem problem;

    std::map<std::string, std::vector<RectifyCost* > > cost_functions;

    double rot_vec[3] = {0.001, 0.001, -0.001};

    std::vector<std::vector<double> > rot_vecs(rows*cols, std::vector<double>(3, .001));

    size_t counter = 0;
    for (int row = -int(rows/2); row <= int(rows/2); ++row) {
        for (int col = -int(cols/2); col <= int(cols/2); ++col, ++counter) {
            clog::L(__func__, 2) << "row: " << row << ", col: " << col << ", counter: " << counter << ", name: " << images[counter] << std::endl;
            size_t next_counter = counter+1;
            if (col == int(cols/2)) {
                next_counter = counter-1;
            }
            CornerStore const& current = data[images[counter]];
            size_t const current_id = getId(images[counter]);
            CornerStore const& next = data[images[next_counter]];
            size_t const next_id = getId(images[next_counter]);
            std::vector<RectifyCost* > & target_costs = cost_functions[images[counter]];

            addImagePairToRectificationProblem(
                        calib,
                        current,
                        current_id,
                        next,
                        next_id,
                        target_costs,
                        problem,
                        0,
                        rot_vecs[counter].data());
        }
    }
    counter = 0;
    for (int row = -int(rows/2); row <= int(rows/2); ++row) {
        for (int col = -int(cols/2); col <= int(cols/2); ++col, ++counter) {
            clog::L(__func__, 2) << "row: " << row << ", col: " << col << ", counter: " << counter << ", name: " << images[counter] << std::endl;
            size_t next_counter = counter+cols;
            if (row == int(rows/2)) {
                next_counter = counter-cols;
            }
            CornerStore const& current = data[images[counter]];
            size_t const current_id = getId(images[counter]);
            CornerStore const& next = data[images[next_counter]];
            size_t const next_id = getId(images[next_counter]);
            std::vector<RectifyCost* > & target_costs = cost_functions[images[counter]];

            addImagePairToRectificationProblem(
                        calib,
                        current,
                        current_id,
                        next,
                        next_id,
                        target_costs,
                        problem,
                        1,
                        rot_vecs[counter].data());
        }
    }
    ceres::Solver::Options options;
    options.num_threads = int(threads);
    options.max_num_iterations = 150;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.function_tolerance = ceres_tolerance;
    options.gradient_tolerance = ceres_tolerance;
    options.parameter_tolerance = ceres_tolerance;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    clog::L(__func__, 1) << summary.FullReport() << "\n";



    /* Global residual statistics */
    runningstats::QuantileStats<double> g_res[2];
    runningstats::QuantileStats<double> g_err[2];
    counter = 0;
    for (const auto& it : cost_functions) {
        /* Local residual statistics */
        runningstats::QuantileStats<double> res[2];
        runningstats::QuantileStats<double> err[2];

        for (const auto cost : it.second) {
            double residuals[1];
            cost->operator()(rot_vec, residuals);
            g_res[cost->axis].push_unsafe(residuals[0]);
            res[cost->axis].push_unsafe(residuals[0]);
            g_err[cost->axis].push_unsafe(std::abs(residuals[0]));
            err[cost->axis].push_unsafe(std::abs(residuals[0]));
        }

        clog::L(__func__, 2) << "Residual stats for " << it.first << ":" << std::endl;
        for (size_t ii = 0; ii < 2; ++ii) {
            clog::L(__func__, 2) << res[ii].print() << std::endl;
            res[ii].plotHistAndCDF(std::string("individual-rect-local-residuals-")
                                   + std::to_string(counter) + "-" + std::to_string(ii), .1);
        }
        clog::L(__func__, 2) << "Error stats for " << it.first << ":" << std::endl;
        for (size_t ii = 0; ii < 2; ++ii) {
            clog::L(__func__, 2) << err[ii].print() << std::endl;
        }
        counter++;
    }
    clog::L(__func__, 1) << "Residual stats for all images:" << std::endl;
    for (size_t ii = 0; ii < 2; ++ii) {
        clog::L(__func__, 1) << g_res[ii].print() << std::endl;
        g_res[ii].plotHistAndCDF(std::string("individual-rect-global-residuals-") + std::to_string(ii), .1);
    }
    clog::L(__func__, 1) << "Error stats for all images:" << std::endl;
    for (size_t ii = 0; ii < 2; ++ii) {
        clog::L(__func__, 1) << g_err[ii].print() << std::endl;
    }
    clog::L(__func__, 1) << std::endl;

    clog::L(__func__, 1) << "Rotation vectors: " << std::endl;
    for (size_t ii = 0; ii < rot_vecs.size(); ++ii) {
        for (size_t jj = 0; jj < 3; ++jj) {
            clog::L(__func__, 1) << rot_vecs[ii][jj] << ", ";
        }
        clog::L(__func__, 1) << std::endl;
    }
}


} // namespace hdcalib
