#include "hdcalib.h"

namespace hdcalib {


struct RectifyCost {
    int8_t const axis;
    cv::Vec3d const src_a;
    cv::Vec3d const src_b;

    cv::Mat_<double> const& cameraMatrix;

    RectifyCost(int8_t const _axis,
                cv::Vec3d const _src_a,
                cv::Vec3d const _src_b,
                cv::Mat_<double> const& _cameraMatrix) :
        axis(_axis),
        src_a(_src_a),
        src_b(_src_b),
    cameraMatrix(_cameraMatrix) {}

    template<class T>
    void compute(
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
    bool operator () (
            T const * const rot_vec,
            T * residuals) const {


        T result1[2] = {T(0), T(0)};
        T result2[2] = {T(0), T(0)};

        compute(rot_vec, result1, result2);

        residuals[0] = result1[1-axis] - result2[1-axis];

        return true;
    }
};

template<class RCOST>
void Calib::addImagePairToRectificationProblem(
        CornerStore const& current,
        size_t const current_id,
        CornerStore const& next,
        size_t const next_id,
        std::vector<RCOST* > & target_costs,
        ceres::Problem & problem,
        int8_t const axis,
        double rot_vec[3]
        ) {
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
        src1 = get3DPoint(src, rvecs[current_id], tvecs[current_id]);
        dst1 = get3DPoint(dst, rvecs[next_id], tvecs[next_id]);
        RectifyCost * cost = new RectifyCost(axis, src1, dst1, cameraMatrix);
        ceres::CostFunction * cost_function = new ceres::AutoDiffCostFunction<RectifyCost, 1, 3>(
                    cost
                    );
        //ceres::LossFunction * loss_func = new ceres::CauchyLoss(10);
        ceres::LossFunction * loss_func = nullptr;

        problem.AddResidualBlock(cost_function, loss_func, rot_vec);
        target_costs.push_back(cost);
    }
}

void Calib::getRectificationRotation(const size_t rows, const size_t cols, const std::vector<std::string> &images, cv::Vec3d &rect_rot) {
    std::cout << "##### getRectificationRotation #####" << std::endl;
    prepareCalibration();
    ceres::Problem problem;

    std::map<std::string, std::vector<RectifyCost* > > cost_functions;

    double rot_vec[3] = {0.1, 0.1, -0.1};

    size_t counter = 0;
    for (int row = -int(rows/2); row <= int(rows/2); ++row) {
        for (int col = -int(cols/2); col <= int(cols/2); ++col, ++counter) {
            std::cout << "row: " << row << ", col: " << col << ", counter: " << counter << ", name: " << images[counter] << std::endl;
            if (col == int(cols/2)) {
                continue;
            }
            CornerStore const& current = data[images[counter]];
            size_t const current_id = getId(images[counter]);
            CornerStore const& next = data[images[counter+1]];
            size_t const next_id = getId(images[counter+1]);
            std::vector<RectifyCost* > & target_costs = cost_functions[images[counter]];

            addImagePairToRectificationProblem(
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
            std::cout << "row: " << row << ", col: " << col << ", counter: " << counter << ", name: " << images[counter] << std::endl;
            if (row == int(rows/2)) {
                continue;
            }
            CornerStore const& current = data[images[counter]];
            size_t const current_id = getId(images[counter]);
            CornerStore const& next = data[images[counter+cols]];
            size_t const next_id = getId(images[counter+cols]);
            std::vector<RectifyCost* > & target_costs = cost_functions[images[counter]];

            addImagePairToRectificationProblem(
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
    ceres::Solver::Options options;
    options.num_threads = threads;
    options.max_num_iterations = 150;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    options.function_tolerance = 1e-16;
    options.gradient_tolerance = 1e-16;
    options.parameter_tolerance = 1e-16;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << "\n";

    /* Global residual statistics */
    runningstats::QuantileStats<double> g_res[2];
    runningstats::QuantileStats<double> g_err[2];
    counter = 0;

    std::ofstream global_out_1("rect-values1.data");
    global_out_1 << "#Residuals for the x-axis 1-2. x/y value of [estimated 3D-position of marker projected to image plance]. 3-4. same for second value. 5.: residual" << std::endl;
    std::ofstream global_out_2("rect-values2.data");
    global_out_2 << "#Residuals for the y-axis 1-2. x/y value of [estimated 3D-position of marker projected to image plance]. 3-4. same for second value. 5.: residual" << std::endl;
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
            double result1[2] = {0,0};
            double result2[2] = {0,0};
            cost->compute(rot_vec, result1, result2);
            if (0 == cost->axis) {
                global_out_1 << result1[0] << "\t" << result1[1] << "\t" << result2[0] << "\t" << result2[1] << "\t" << residuals[0] << std::endl;
            }
            else {
                global_out_2 << result1[0] << "\t" << result1[1] << "\t" << result2[0] << "\t" << result2[1] << "\t" << residuals[0] << std::endl;
            }
        }

        std::cout << "Residual stats for " << it.first << ":" << std::endl;
        for (size_t ii = 0; ii < 2; ++ii) {
            std::cout << res[ii].print() << std::endl;
            res[ii].plotHistAndCDF(std::string("rect-local-residuals-")
                             + std::to_string(counter) + "-" + std::to_string(ii), .1);
        }
        std::cout << "Error stats for " << it.first << ":" << std::endl;
        for (size_t ii = 0; ii < 2; ++ii) {
            std::cout << err[ii].print() << std::endl;
        }
        std::cout << std::endl;
        counter++;
    }
    std::cout << "Residual stats for all images:" << std::endl;
    for (size_t ii = 0; ii < 2; ++ii) {
        std::cout << g_res[ii].print() << std::endl;
        g_res[ii].plotHistAndCDF(std::string("rect-global-residuals-") + std::to_string(ii), .1);
    }
    std::cout << "Error stats for all images:" << std::endl;
    for (size_t ii = 0; ii < 2; ++ii) {
        std::cout << g_err[ii].print() << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Rotation vector: " << rot_vec[0] << ", " << rot_vec[1] << ", " << rot_vec[2] << std::endl;
}

void Calib::getIndividualRectificationRotation(const size_t rows, const size_t cols, const std::vector<std::string> &images, cv::Vec3d &rect_rot) {
    std::cout << "##### getRectificationRotation #####" << std::endl;
    prepareCalibration();
    ceres::Problem problem;

    std::map<std::string, std::vector<RectifyCost* > > cost_functions;

    double rot_vec[3] = {0.0, 0.0, 0.1};

    std::vector<std::vector<double> > rot_vecs(rows*cols, std::vector<double>(3, .001));

    size_t counter = 0;
    for (int row = -int(rows/2); row <= int(rows/2); ++row) {
        for (int col = -int(cols/2); col <= int(cols/2); ++col, ++counter) {
            std::cout << "row: " << row << ", col: " << col << ", counter: " << counter << ", name: " << images[counter] << std::endl;
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
            std::cout << "row: " << row << ", col: " << col << ", counter: " << counter << ", name: " << images[counter] << std::endl;
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
    options.num_threads = threads;
    options.max_num_iterations = 150;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.function_tolerance = 1e-16;
    options.gradient_tolerance = 1e-16;
    options.parameter_tolerance = 1e-16;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << "\n";

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

        std::cout << "Residual stats for " << it.first << ":" << std::endl;
        for (size_t ii = 0; ii < 2; ++ii) {
            std::cout << res[ii].print() << std::endl;
            res[ii].plotHistAndCDF(std::string("individual-rect-local-residuals-")
                             + std::to_string(counter) + "-" + std::to_string(ii), .1);
        }
        std::cout << "Error stats for " << it.first << ":" << std::endl;
        for (size_t ii = 0; ii < 2; ++ii) {
            std::cout << err[ii].print() << std::endl;
        }
        std::cout << std::endl;
        counter++;
    }
    std::cout << "Residual stats for all images:" << std::endl;
    for (size_t ii = 0; ii < 2; ++ii) {
        std::cout << g_res[ii].print() << std::endl;
        g_res[ii].plotHistAndCDF(std::string("individual-rect-global-residuals-") + std::to_string(ii), .1);
    }
    std::cout << "Error stats for all images:" << std::endl;
    for (size_t ii = 0; ii < 2; ++ii) {
        std::cout << g_err[ii].print() << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Rotation vectors: " << std::endl;
    for (size_t ii = 0; ii < rot_vecs.size(); ++ii) {
        for (size_t jj = 0; jj < 3; ++jj) {
            std::cout << rot_vecs[ii][jj] << ", ";
        }
        std::cout << std::endl;
    }
}

} // namespace hdcalib
