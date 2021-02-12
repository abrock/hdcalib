#include "hdcalib.h"

#undef NDEBUG
#include <assert.h>

namespace  {
template<class T>
void vec2arr(T arr[3], cv::Point3d const& p) {
    arr[0] = p.x;
    arr[1] = p.y;
    arr[2] = p.z;
}

template<class T>
void vec2arr(T arr[2], cv::Point2d const& p) {
    arr[0] = p.x;
    arr[1] = p.y;
}
}

namespace hdcalib {

void Calib::prepareCalibration() {
    if (preparedCalib && imagePoints.size() == data.size() && objectPoints.size() == data.size()) {
        return;
    }
    preparedOpenCVCalib = false;
    preparedCalib = true;

    imagePoints = std::vector<std::vector<cv::Point2f> >(data.size());
    objectPoints = std::vector<std::vector<cv::Point3f> >(data.size());
    imageFiles.resize(data.size());

    size_t ii = 0;
    for (std::pair<const std::string, CornerStore> const& it : data) {
        it.second.getPoints(imagePoints[ii], objectPoints[ii], *this);
        imageFiles[ii] = it.first;
        ++ii;
    }
}

void CornerStore::getPoints(
        std::vector<Point2f> &imagePoints,
        std::vector<Point3f> &objectPoints,
        hdcalib::Calib const& calib) const {
    imagePoints.resize(size());
    objectPoints.resize(size());
    for (size_t ii = 0; ii < size(); ++ii) {
        hdmarker::Corner const& c = get(ii);
        imagePoints[ii] = (c.p);
        objectPoints[ii] = calib.getInitial3DCoord(c);
    }
}

double Calib::CeresCalib(double const outlier_threshold) {
    // Build the problem.
    ceres::Problem problem;

    prepareCalibration();

    std::vector<std::vector<double> > local_rvecs(data.size()), local_tvecs(data.size());

    if (!hasCalibName("Ceres")) {
        if (hasCalibName("OpenCV")) {
            calibrations["Ceres"] = calibrations["OpenCV"];
        }
        else if (hasCalibName("SimpleCeres")) {
            calibrations["Ceres"] = calibrations["SimpleCeres"];
        }
        else if (hasCalibName("SimpleOpenCV")) {
            calibrations["Ceres"] = calibrations["SimpleOpenCV"];
        }
    }

    CalibResult & calib = calibrations["Ceres"];

    assert(calib.rvecs.size() == imageFiles.size());
    assert(calib.tvecs.size() == imageFiles.size());


    std::vector<double> local_dist = mat2vec(calib.distCoeffs);

    cv::Mat_<double> old_cam = calib.cameraMatrix.clone();
    cv::Mat_<double> old_dist = calib.distCoeffs.clone();

    if (calib.outlier_percentages.size() != data.size()) {
        calib.outlier_percentages = std::vector<double>(data.size(), 0.0);
    }
    std::multimap<double, std::string> outlier_ranking;
    runningstats::RunningStats outlier_stats;
    size_t ignored_files_counter = 0;
    for (size_t image_index = 0; image_index < data.size(); ++image_index) {
        local_rvecs[image_index] = mat2vec(calib.rvecs[image_index]);
        local_tvecs[image_index] = mat2vec(calib.tvecs[image_index]);

        std::vector<cv::Point2d> markers, reprojections;
        getReprojections(calib, image_index, markers, reprojections);
        std::vector<cv::Point2f> local_image_points;
        std::vector<cv::Point3f> local_object_points;
        if (outlier_threshold < 0) {
            local_image_points = imagePoints[image_index];
            local_object_points = objectPoints[image_index];
        }
        else {
            for (size_t ii = 0; ii < markers.size() && ii < reprojections.size(); ++ii) {
                cv::Point2d const& marker = markers[ii];
                cv::Point2d const& reprojection = reprojections[ii];
                double const error = distance(marker, reprojection);
                if (error < outlier_threshold) {
                    local_image_points.push_back(imagePoints[image_index][ii]);
                    local_object_points.push_back(objectPoints[image_index][ii]);
                }
            }
            double const outlier_percentage = 100.0 * double(imagePoints[image_index].size() - local_image_points.size()) / imagePoints[image_index].size();
            outlier_stats.push_unsafe(outlier_percentage);
            outlier_ranking.insert({outlier_percentage, imageFiles[image_index]});
            calib.outlier_percentages[image_index] = outlier_percentage;
            if (outlier_percentage > max_outlier_percentage) {
                ignored_files_counter++;
                continue;
            }
        }

        ceres::CostFunction * cost_function =
                new ceres::AutoDiffCostFunction<
                ProjectionFunctor,
                ceres::DYNAMIC,
                1, // focal length x
                1, // focal length y
                1, // principal point x
                1, // principal point y
                3, // rotation vector for the target
                3, // translation vector for the target
                14 // distortion coefficients

                >(new ProjectionFunctor(local_image_points, local_object_points),
                  int(2*local_image_points.size()) // Number of residuals
                  );
        problem.AddResidualBlock(cost_function,
                                 cauchy_param > 0 ? new ceres::CauchyLoss(cauchy_param) : nullptr, // Loss function (nullptr = L2)
                                 &calib.cameraMatrix(0,0), // focal length x
                                 &calib.cameraMatrix(1,1), // focal length y
                                 &calib.cameraMatrix(0,2), // principal point x
                                 &calib.cameraMatrix(1,2), // principal point y
                                 local_rvecs[image_index].data(), // rotation vector for the target
                                 local_tvecs[image_index].data(), // translation vector for the target
                                 local_dist.data() // distortion coefficients
                                 );

    }
    clog::L(__func__, 2) << "Outlier ranking:" << std::endl;
    for (auto const& it : outlier_ranking) {
        clog::L(__func__, 2) << it.second << ": \t" << it.first << std::endl;
    }
    if (outlier_threshold > 0) {
        clog::L(__func__, 1) << "Outlier percentage per image stats: " << outlier_stats.print() << std::endl;
    }
    std::cout << "Ignored files: " << 100.0 * double(ignored_files_counter) / data.size() << "%" << std::endl;


    // Run the solver!
    ceres::Solver::Options options;
    options.num_threads = int(threads);
    options.max_num_iterations = 150;
    options.function_tolerance = ceres_tolerance;
    options.gradient_tolerance = ceres_tolerance;
    options.parameter_tolerance = ceres_tolerance;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    clog::L(__func__, 1) << summary.BriefReport() << "\n";
    clog::L(__func__, 1) << summary.FullReport() << "\n";

    size_t counter = 0;
    for (auto & it : calib.distCoeffs) {
        it = local_dist[counter];
        counter++;
    }

    calib.rvecs.resize(local_rvecs.size());
    calib.tvecs.resize(local_rvecs.size());
    for (size_t ii = 0; ii < local_rvecs.size(); ++ii) {
        calib.rvecs[ii] = vec2mat(local_rvecs[ii]);
        calib.tvecs[ii] = vec2mat(local_tvecs[ii]);
    }
    calib.imageFiles = imageFiles;

    clog::L(__func__, 1) << "Parameters before: " << std::endl
                         << "Camera matrix: " << old_cam << std::endl
                         << "Distortion: " << old_dist << std::endl;
    clog::L(__func__, 1) << "Parameters after: " << std::endl
                         << "Camera matrix: " << calib.cameraMatrix << std::endl
                         << "Distortion: " << calib.distCoeffs << std::endl;
    clog::L(__func__, 1) << "Difference: old - new" << std::endl
                         << "Camera matrix: " << (old_cam - calib.cameraMatrix) << std::endl
                         << "Distortion: " << (old_dist - calib.distCoeffs) << std::endl;

    hasCalibration = true;

    return 0;
}

double Calib::SimpleCeresCalib(const double outlier_threshold) {
    // Build the problem.
    ceres::Problem problem;

    prepareCalibration();

    std::vector<std::vector<double> > local_rvecs(data.size()), local_tvecs(data.size());

    if (!hasCalibName("SimpleCeres")) {
        if (hasCalibName("SimpleOpenCV")) {
            calibrations["SimpleCeres"] = calibrations["SimpleOpenCV"];
        }
    }

    CalibResult & calib = calibrations["SimpleCeres"];

    cv::Mat_<double> old_cam = calib.cameraMatrix.clone();

    cv::Point2f const principal(double(imageSize.width-1)/2, double(imageSize.height-1)/2);

    for (size_t ii = 0; ii < data.size(); ++ii) {
        local_rvecs[ii] = mat2vec(calib.rvecs[ii]);
        local_tvecs[ii] = mat2vec(calib.tvecs[ii]);

        ceres::CostFunction * cost_function =
                new ceres::AutoDiffCostFunction<
                SimpleProjectionFunctor,
                ceres::DYNAMIC,
                1, // focal length f
                3, // rotation vector for the target
                3 // translation vector for the target
                >(new SimpleProjectionFunctor(imagePoints[ii], objectPoints[ii], principal),
                  int(2*imagePoints[ii].size()) // Number of residuals
                  );
        problem.AddResidualBlock(cost_function,
                                 nullptr, // Loss function (nullptr = L2)
                                 &calib.cameraMatrix(0,0), // focal length x
                                 local_rvecs[ii].data(), // rotation vector for the target
                                 local_tvecs[ii].data() // translation vector for the target
                                 );

    }


    // Run the solver!
    ceres::Solver::Options options;
    options.num_threads = int(threads);
    options.max_num_iterations = 150;
    options.function_tolerance = ceres_tolerance;
    options.gradient_tolerance = ceres_tolerance;
    options.parameter_tolerance = ceres_tolerance;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    clog::L(__func__, 1) << summary.BriefReport() << "\n";
    clog::L(__func__, 1) << summary.FullReport() << "\n";

    calib.rvecs.resize(local_rvecs.size());
    calib.tvecs.resize(local_rvecs.size());
    for (size_t ii = 0; ii < local_rvecs.size(); ++ii) {
        calib.rvecs[ii] = vec2mat(local_rvecs[ii]);
        calib.tvecs[ii] = vec2mat(local_tvecs[ii]);
    }
    calib.cameraMatrix(1,1) = calib.cameraMatrix(0,0);
    calib.imageFiles = imageFiles;

    clog::L(__func__, 1) << "Parameters before: " << std::endl
                         << "Camera matrix: " << old_cam << std::endl;
    clog::L(__func__, 1) << "Parameters after: " << std::endl
                         << "Camera matrix: " << calib.cameraMatrix << std::endl
                         << "Distortion: " << calib.distCoeffs << std::endl;
    clog::L(__func__, 1) << "Difference: old - new" << std::endl
                         << "Camera matrix: " << (old_cam - calib.cameraMatrix) << std::endl;

    hasCalibration = true;

    return 0;
}

void Calib::setCeresTolerance(const double new_tol) {
    ceres_tolerance = std::abs(new_tol);
}

struct LocalCorrectionsSum {
    /**
    std::map<cv::Point3i, std::vector<double>, cmpSimpleIndex3<cv::Point3i> > const& local_corrections;
    LocalCorrectionsSum(std::map<cv::Point3i, std::vector<double>, cmpSimpleIndex3<cv::Point3i> > const& _local_corrections) :
        local_corrections(_local_corrections) {}
    **/

    LocalCorrectionsSum() {}
    template<class T>
    bool operator () (
            T const * const correction,
            T *residuals) const {
        for (size_t ii = 0; ii < 3; ++ii) {
            residuals[ii] = correction[ii];
        }
        return true;
    }
};

double Calib::CeresCalibFlexibleTarget(double const outlier_threshold) {
    // Build the problem.
    ceres::Problem problem;

    prepareCalibration();

    if (!hasCalibName("Flexible")) {
        if (hasCalibName("Ceres")) {
            calibrations["Flexible"] = calibrations["Ceres"];
        }
        else if (hasCalibName("OpenCV")) {
            calibrations["Flexible"] = calibrations["OpenCV"];
        }
    }

    CalibResult & calib = calibrations["Flexible"];

    std::vector<std::vector<double> > local_rvecs(data.size()), local_tvecs(data.size());

    std::vector<double> local_dist = mat2vec(calib.distCoeffs);

    cv::Mat_<double> old_cam = calib.cameraMatrix.clone();
    cv::Mat_<double> old_dist = calib.distCoeffs.clone();


    std::map<cv::Scalar_<int>, std::vector<double>, cmpScalar > local_corrections;

    for (const auto& it: data) {
        for (size_t ii = 0; ii < it.second.size(); ++ii) {
            cv::Scalar_<int> const c = getSimpleIdLayer(it.second.get(ii));
            local_corrections[c] = point2vec3f(calib.objectPointCorrections[c]);
        }
    }

    std::set<cv::Scalar_<int>, cmpScalar> ids;

    if (calib.outlier_percentages.size() != data.size()) {
        calib.outlier_percentages = std::vector<double>(data.size(), 0.0);
    }
    size_t ignored_files_counter = 0;
    std::multimap<double, std::string> outlier_ranking;
    runningstats::RunningStats outlier_percentages;
    for (size_t ii = 0; ii < data.size(); ++ii) {
        local_rvecs[ii] = mat2vec(calib.rvecs[ii]);
        local_tvecs[ii] = mat2vec(calib.tvecs[ii]);
        bool ignore_current_file = false;
        if (calib.outlier_percentages[ii] > max_outlier_percentage) {
            ignore_current_file = true;
            ignored_files_counter++;
        }

        auto const & sub_data = data[imageFiles[ii]];
        size_t outlier_counter = 0;
        for (size_t jj = 0; jj < sub_data.size(); ++jj) {
            cv::Scalar_<int> const c = getSimpleIdLayer(sub_data.get(jj));
            ids.insert(c);
            {
                FlexibleTargetProjectionFunctor loss (
                            imagePoints[ii][jj],
                            objectPoints[ii][jj]
                            );
                double residuals[2] = {0,0};
                if (outlier_threshold > 0) {
                    loss(&calib.cameraMatrix(0,0), // focal length x
                         &calib.cameraMatrix(1,1), // focal length y
                         &calib.cameraMatrix(0,2), // principal point x
                         &calib.cameraMatrix(1,2), // principal point y
                         local_rvecs[ii].data(), // rotation vector for the target
                         local_tvecs[ii].data(), // translation vector for the target
                         local_corrections[c].data(),
                         local_dist.data(),
                         residuals);
                    if (residuals[0] * residuals[0] + residuals[1] * residuals[1] > outlier_threshold * outlier_threshold) {
                        outlier_counter++;
                        continue;
                    }
                }
                if (ignore_current_file) {
                    continue;
                }
                ceres::CostFunction * cost_function =
                        new ceres::AutoDiffCostFunction<
                        FlexibleTargetProjectionFunctor,
                        2, // Number of residuals
                        1, // focal length x
                        1, // focal length y
                        1, // principal point x
                        1, // principal point y
                        3, // rotation vector for the target
                        3, // translation vector for the target
                        3, // correction vector for the 3d marker position
                        14 // distortion coefficients

                        >(new FlexibleTargetProjectionFunctor (
                              imagePoints[ii][jj],
                              objectPoints[ii][jj]
                              ));
                problem.AddResidualBlock(cost_function,
                                         cauchy_param > 0 ? new ceres::CauchyLoss(cauchy_param) : nullptr, // Loss function (nullptr = L2)
                                         &calib.cameraMatrix(0,0), // focal length x
                                         &calib.cameraMatrix(1,1), // focal length y
                                         &calib.cameraMatrix(0,2), // principal point x
                                         &calib.cameraMatrix(1,2), // principal point y
                                         local_rvecs[ii].data(), // rotation vector for the target
                                         local_tvecs[ii].data(), // translation vector for the target
                                         local_corrections[c].data(),
                                         local_dist.data() // distortion coefficients
                                         );
            }
        }
        double const outlier_percentage = 100.0 * double(outlier_counter) / sub_data.size();
        outlier_percentages.push_unsafe(outlier_percentage);
        outlier_ranking.insert({outlier_percentage, imageFiles[ii]});
        calib.outlier_percentages[ii] = outlier_percentage;
    }
    clog::L(__func__, 2) << "Outlier ranking:" << std::endl;
    for (auto const& it : outlier_ranking) {
        clog::L(__func__, 2) << it.second << ": \t" << it.first << std::endl;
    }
    std::cout << "Ignored " << 100.0 * double(ignored_files_counter) / data.size() << "% of files" << std::endl;
    if (outlier_threshold > 0) {
        clog::L(__func__, 2) << "Outlier percentage stats: " << outlier_percentages.print() << std::endl;
    }

    for (cv::Scalar_<int> const& it : ids) {
        ceres::CostFunction* cost_function =
                new ceres::AutoDiffCostFunction<LocalCorrectionsSum, 3, 3>(
                    new LocalCorrectionsSum());
        problem.AddResidualBlock(cost_function,
                                 nullptr, // Loss function (nullptr = L2)
                                 local_corrections[it].data() // correction
                                 );
    }

    // Run the solver!
    ceres::Solver::Options options;
    options.num_threads = int(threads);
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.max_num_iterations = 150;
    options.minimizer_progress_to_stdout = verbose;
    options.function_tolerance = ceres_tolerance;
    options.gradient_tolerance = ceres_tolerance;
    options.parameter_tolerance = ceres_tolerance;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    clog::L(__func__, 1) << summary.BriefReport() << "\n";
    clog::L(__func__, 1) << summary.FullReport() << "\n";

    size_t counter = 0;
    for (auto & it : calib.distCoeffs) {
        it = local_dist[counter];
        counter++;
    }

    for (auto const& it : local_corrections) {
        calib.objectPointCorrections[it.first] = vec2point3f(it.second);
    }

    for (size_t ii = 0; ii < local_rvecs.size(); ++ii) {
        calib.rvecs[ii] = vec2mat(local_rvecs[ii]);
        calib.tvecs[ii] = vec2mat(local_tvecs[ii]);
    }

    clog::L(__func__, 1) << "Parameters before: " << std::endl
                         << "Camera matrix: " << old_cam << std::endl
                         << "Distortion: " << old_dist << std::endl;
    clog::L(__func__, 1) << "Parameters after: " << std::endl
                         << "Camera matrix: " << calib.cameraMatrix << std::endl
                         << "Distortion: " << calib.distCoeffs << std::endl;
    clog::L(__func__, 1) << "Difference: old - new" << std::endl
                         << "Camera matrix: " << (old_cam - calib.cameraMatrix) << std::endl
                         << "Distortion: " << (old_dist - calib.distCoeffs) << std::endl;

    return 0;
}

template<class F, class T>
void Calib::project(
        F const p[3],
T result[2],
const T focal[2],
const T principal[2],
const T R[9],
const T t[3]
) {
    T const X(p[0]), Y(p[1]), Z(p[2]);
    T& x = result[0];
    T& y = result[1];
    T z;
    z = R[6]*X + R[7]*Y + R[8]*Z + t[2];
    if (std::numeric_limits<double>::min() > ceres::abs(z)) {
        z = T(1);
    }
    x = (R[0]*X + R[1]*Y + R[2]*Z + t[0])/z;
    y = (R[3]*X + R[4]*Y + R[5]*Z + t[1])/z;

    x = x * focal[0] + principal[0];
    y = y * focal[1] + principal[1];
}

template void Calib::project(
double const p[3],
double result[2],
const double focal[2],
const double principal[2],
const double R[9],
const double t[3]
);

template void Calib::project(
ceres::Jet<double, 3> const p[3],
ceres::Jet<double, 3> result[2],
const ceres::Jet<double, 3> focal[2],
const ceres::Jet<double, 3> principal[2],
const ceres::Jet<double, 3> R[9],
const ceres::Jet<double, 3> t[3]
);

template<typename T>
void applySensorTilt(
        T& x, T& y,
        T const& tau_x, T const& tau_y
        ) {
    T const s_x = ceres::sin(tau_x);
    T const s_y = ceres::sin(tau_y);
    T const c_x = ceres::cos(tau_x);
    T const c_y = ceres::cos(tau_y);

    T const x1 = c_y*x + s_x*s_y*y - s_y*c_x;
    T const y1 = c_x*y + s_x;
    T const z1 = s_y*x - c_y*s_x*y + c_y*c_x;

    x = (c_y*c_x*x1 + s_y*c_x*z1)/z1;
    y = (c_y*c_x*y1 - s_x*z1)/z1;
}

template<class F, class T>
void Calib::project(
        F const p[3],
T result[2],
const T focal[2],
const T principal[2],
const T R[9],
const T t[3],
const T dist[14]
) {
    T const X(p[0]), Y(p[1]), Z(p[2]);
    T& x = result[0];
    T& y = result[1];
    T z;
    z = R[6]*X + R[7]*Y + R[8]*Z + t[2];
    if (std::numeric_limits<double>::min() > ceres::abs(z)) {
        z = T(1);
    }
    x = (R[0]*X + R[1]*Y + R[2]*Z + t[0])/z;
    y = (R[3]*X + R[4]*Y + R[5]*Z + t[1])/z;

    //(k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]])
    T const& k1 = dist[0];
    T const& k2 = dist[1];
    T const& p1 = dist[2];
    T const& p2 = dist[3];
    T const& k3 = dist[4];
    T const& k4 = dist[5];
    T const& k5 = dist[6];
    T const& k6 = dist[7];
    T const& s1 = dist[8];
    T const& s2 = dist[9];
    T const& s3 = dist[10];
    T const& s4 = dist[11];
    T const& tau_x = dist[12];
    T const& tau_y = dist[13];

    T const r2 = x*x + y*y;
    T const r4 = r2*r2;
    T const r6 = r4*r2;

    T x2 = x*(T(1) + k1*r2 + k2*r4 + k3*r6)/(T(1) + k4*r2 + k5*r4 + k6*r6)
            + T(2)*x*y*p1 + p2*(r2 + T(2)*x*x) + s1*r2 + s2*r4;

    T y2 = y*(T(1) + k1*r2 + k2*r4 + k3*r6)/(T(1) + k4*r2 + k5*r4 + k6*r6)
            + T(2)*x*y*p2 + p1*(r2 + T(2)*y*y) + s3*r2 + s4*r4;

    applySensorTilt(x2, y2, tau_x, tau_y);

    x = x2 * focal[0] + principal[0];
    y = y2 * focal[1] + principal[1];
}

template void Calib::project(
double const p[3],
double result[2],
const double focal[2],
const double principal[2],
const double R[9],
const double t[3],
const double dist[14]
);

template<class T>
void Calib::rot_vec2mat(const T vec[], T mat[]) {
    T const theta = ceres::sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
    T const c = ceres::cos(theta);
    T const s = ceres::sin(theta);
    T const c1 = T(1) - c;

    // Calculate normalized vector.
    T const factor = (ceres::abs(theta) < std::numeric_limits<double>::epsilon() ? T(1) : T(1)/theta);
    T const vec_norm[3] = {factor * vec[0], factor * vec[1], factor * vec[2]};

    mat[0] = c + c1*vec_norm[0]*vec_norm[0];
    mat[1] = c1*vec_norm[0]*vec_norm[1] - s*vec_norm[2];
    mat[2] = c1*vec_norm[0]*vec_norm[2] + s*vec_norm[1];

    mat[3] = c1*vec_norm[0]*vec_norm[1] + s*vec_norm[2];
    mat[4] = c + c1*vec_norm[1]*vec_norm[1];
    mat[5] = c1*vec_norm[1]*vec_norm[2] - s*vec_norm[0];

    mat[6] = c1*vec_norm[0]*vec_norm[2] - s*vec_norm[1];
    mat[7] = c1*vec_norm[1]*vec_norm[2] + s*vec_norm[0];
    mat[8] = c + c1*vec_norm[2]*vec_norm[2];
}

template void Calib::rot_vec2mat(const double vec[], double mat[]);
template void Calib::rot_vec2mat(const ceres::Jet<double, 3> vec[], ceres::Jet<double, 3> mat[]);

template<class T, class U>
void Calib::normalize_rot4(T const in[4], U out[4]) {
    T sum(0);
    for (size_t ii = 0; ii < 3; ++ii) {
        sum += in[ii] * in[ii];
    }
    if (ceres::abs(sum) < 100*std::numeric_limits<double>::min()) {
        out[0] = out[1] = out[3] = T(0);
        out[2] = T(1);
        return;
    }
    T factor(1.0/std::sqrt(sum));
    if (in[3] < T(0)) {
        out[3] = in[3]*T(-1);
        factor *= T(-1);
    }
    else {
        out[3] = in[3];
    }
    for (size_t ii = 0; ii < 3; ++ii) {
        out[ii] = factor * in[ii];
    }
}

template void Calib::normalize_rot4(double const [4], double[4]);

template<class T>
void Calib::rot4_vec2mat(const T vec[], T mat[]) {
    T const vec_length = ceres::sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
    if (ceres::abs(vec_length) < 100 * std::numeric_limits<double>::min()) {
        for (size_t ii = 0; ii < 9; ++ii) {
            mat[ii] = T(0);
        }
        mat[0] = mat[4] = mat[8] = T(1);
        return;
    }

    T const c = ceres::cos(vec[3]);
    T const s = ceres::sin(vec[3]);
    T c1 = T(1) - c;

    // Calculate normalized vector.
    T const factor = (ceres::abs(vec_length) < 100*std::numeric_limits<double>::min() ? T(1) : T(1)/vec_length);
    T const vec_norm[3] = {factor * vec[0], factor * vec[1], factor * vec[2]};

    mat[0] = c + c1*vec_norm[0]*vec_norm[0];
    mat[1] = c1*vec_norm[0]*vec_norm[1] - s*vec_norm[2];
    mat[2] = c1*vec_norm[0]*vec_norm[2] + s*vec_norm[1];

    mat[3] = c1*vec_norm[0]*vec_norm[1] + s*vec_norm[2];
    mat[4] = c + c1*vec_norm[1]*vec_norm[1];
    mat[5] = c1*vec_norm[1]*vec_norm[2] - s*vec_norm[0];

    mat[6] = c1*vec_norm[0]*vec_norm[2] - s*vec_norm[1];
    mat[7] = c1*vec_norm[1]*vec_norm[2] + s*vec_norm[0];
    mat[8] = c + c1*vec_norm[2]*vec_norm[2];
}

template void Calib::rot4_vec2mat(const double[], double[]);

template<class T>
void Calib::rot3_rot4(const Mat &rvec, T vec[]) {
    cv::Mat_<double> _rvec(rvec);
    double sum = 0;
    for (int ii = 0; ii < 3; ++ii) {
        sum += _rvec(ii)*_rvec(ii);
    }
    double const length = std::sqrt(sum);
    if (length < 100*std::numeric_limits<T>::min()) {
        vec[0] = vec[1] = vec[3] = 0;
        vec[2] = 1;
    }
    else {
        for (int ii = 0; ii < 3; ++ii) {
            vec[size_t(ii)] = _rvec(ii)/length;
        }
        vec[3] = length;
    }
}


template<class T>
void Calib::rot3_rot4(const T src[], T vec[]) {
    cv::Mat_<double> _src(3,1,0.0);
    for (size_t ii = 0; ii < 3; ++ii) {
        _src(int(ii)) = src[ii];
    }
    rot3_rot4(_src, vec);
}

template void Calib::rot3_rot4(const double [], double[]);

template<class T>
void Calib::rot4_rot3(T const vec[], Mat &rvec) {
    cv::Mat_<double> result(3,1,0.0);
    double sum = 0;
    for (size_t ii = 0; ii < 3; ++ii) {
        sum += vec[ii]*vec[ii];
    }

    if (std::abs(vec[3]) < 100*std::numeric_limits<T>::min()
            || sum < 100*std::numeric_limits<T>::min()) {
        rvec = result;
        return;
    }

    double const factor = std::abs(vec[3]) < 100*std::numeric_limits<T>::min()
            || sum < 100*std::numeric_limits<T>::min()
            ? 0 : vec[3]/std::sqrt(sum);
    for (int ii = 0; ii < 3; ++ii) {
        result(ii) = vec[size_t(ii)] * factor;
    }

    rvec = result;
}

template void Calib::rot4_rot3(double const [], Mat &);

template<class T>
/**
 * @brief rot4_rot3 Converts a (4-entry) rotation vector to a equivalent Rodrigues rotation vector with 3 components.
 * @param rvec
 * @param vec
 */
void Calib::rot4_rot3(T const vec[4], T rvec[3]) {
    cv::Mat_<double> result;
    rot4_rot3(vec, result);
    for (size_t ii = 0; ii < 3; ++ii) {
        rvec[ii] = result(int(ii));
    }
}

template void Calib::rot4_rot3(double const[], double[]);

template<class T>
std::vector<T> Calib::rot3_rot4(cv::Mat const& rvec) {
    std::vector<T> result(4, 0.0);
    rot3_rot4(rvec, result.data());
    return result;
}

template std::vector<double> Calib::rot3_rot4(cv::Mat const& rvec);

ProjectionFunctor::ProjectionFunctor(
        const std::vector<Point2f> &_markers,
        const std::vector<Point3f> &_points) :
    markers(_markers),
    points(_points) {
}

ProjectionFunctor::~ProjectionFunctor() {

}

template<class T>
bool ProjectionFunctor::operator()(
        const T * const f_x,
        const T * const f_y,
        const T * const c_x,
        const T * const c_y,
        const T * const rvec,
        const T * const tvec,
        const T * const dist,
        T *residuals) const {
    T rot_mat[9];
    Calib::rot_vec2mat(rvec, rot_mat);
    T const f[2] = {f_x[0], f_y[0]};
    T const c[2] = {c_x[0], c_y[0]};
    T result[2] = {T(0), T(0)};
    for (size_t ii = 0; ii < markers.size(); ++ii) {
        T point[3] = {T(points[ii].x), T(points[ii].y), T(points[ii].z)};
        Calib::project(point, result, f, c, rot_mat, tvec, dist);
        residuals[2*ii] = result[0] - T(markers[ii].x);
        residuals[2*ii+1] = result[1] - T(markers[ii].y);
    }
    return true;
}

SimpleProjectionFunctor::SimpleProjectionFunctor(
        const std::vector<Point2f> &_markers,
        const std::vector<Point3f> &_points,
        const cv::Point2f & _principal) :
    markers(_markers),
    points(_points),
    principal(_principal) {
}

template<class T>
bool SimpleProjectionFunctor::operator()(
        const T * const focal,
        const T * const rvec,
        const T * const tvec,
        T *residuals) const {
    T rot_mat[9];
    Calib::rot_vec2mat(rvec, rot_mat);
    T const f[2] = {focal[0], focal[0]};
    T const c[2] = {T(principal.x), T(principal.y)};
    T result[2] = {T(0), T(0)};
    T dist[14] = {T(0),T(0),T(0),T(0),T(0),T(0),T(0),T(0),T(0),T(0),T(0),T(0),T(0),T(0)};
    for (size_t ii = 0; ii < markers.size(); ++ii) {
        T point[3] = {T(points[ii].x), T(points[ii].y), T(points[ii].z)};
        Calib::project(point, result, f, c, rot_mat, tvec, dist);
        residuals[2*ii] = result[0] - T(markers[ii].x);
        residuals[2*ii+1] = result[1] - T(markers[ii].y);
    }
    return true;
}

ProjectionFunctorRot4::ProjectionFunctorRot4(
        const std::vector<Point2f> &_markers,
        const std::vector<Point3f> &_points) :
    markers(_markers),
    points(_points) {
}

template<class T>
bool ProjectionFunctorRot4::operator()(
        const T * const f_x,
        const T * const f_y,
        const T * const c_x,
        const T * const c_y,
        const T * const rvec,
        const T * const tvec,
        const T * const dist,
        T *residuals) const {
    T rot_mat[9];
    Calib::rot4_vec2mat(rvec, rot_mat);
    T const f[2] = {f_x[0], f_y[0]};
    T const c[2] = {c_x[0], c_y[0]};
    T result[2] = {T(0), T(0)};
    for (size_t ii = 0; ii < markers.size(); ++ii) {
        T point[3] = {T(points[ii].x), T(points[ii].y), T(points[ii].z)};
        Calib::project(point, result, f, c, rot_mat, tvec, dist);
        residuals[2*ii] = result[0] - T(markers[ii].x);
        residuals[2*ii+1] = result[1] - T(markers[ii].y);
    }
    return true;
}

FlexibleTargetProjectionFunctor::FlexibleTargetProjectionFunctor(
        const Point2f &_marker,
        const Point3f &_point): marker(_marker), point(_point)
{

}

template<class T>
bool FlexibleTargetProjectionFunctor::operator()(
        const T * const f_x,
        const T * const f_y,
        const T * const c_x,
        const T * const c_y,
        const T * const rvec,
        const T * const tvec,
        const T * const correction,
        const T * const dist,
        T *residuals) const
{
    T rot_mat[9];
    Calib::rot_vec2mat(rvec, rot_mat);
    T const f[2] = {f_x[0], f_y[0]};
    T const c[2] = {c_x[0], c_y[0]};
    T result[2] = {T(0), T(0)};
    T const src[3] = {T(point.x) + correction[0],
                      T(point.y) + correction[1],
                      T(point.z) + correction[2]};
    Calib::project(src, result, f, c, rot_mat, tvec, dist);
    residuals[0] = T(weight) * (result[0] - T(marker.x));
    residuals[1] = T(weight) * (result[1] - T(marker.y));
    return true;
}

template<int LENGTH>
VecLengthFunctor<LENGTH>::VecLengthFunctor(const double _target_square_length) : target_square_length(_target_square_length){

}

template<int LENGTH>
template<class T>
bool VecLengthFunctor<LENGTH>::operator()(const T * const vec, T *residual) const {
    residual[0] = T(-target_square_length);
    for (size_t ii = 0; ii < LENGTH; ++ii) {
        residual[0] += vec[ii] * vec[ii];
    }
    return true;
}


} // namespace hdcalib
