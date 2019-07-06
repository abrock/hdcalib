#include "hdcalib.h"

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
    for (std::pair<std::string, CornerStore> const& it : data) {
        it.second.getPoints(imagePoints[ii], objectPoints[ii], *this);
        imageFiles[ii] = it.first;
        ++ii;
    }
}


double Calib::CeresCalib() {
    if (verbose) {
        std::cout << "Running Ceres calibration" << std::endl;
    }

    // Build the problem.
    ceres::Problem problem;

    prepareCalibration();

    std::vector<std::vector<double> > local_rvecs(data.size()), local_tvecs(data.size());

    std::vector<double> local_dist = mat2vec(distCoeffs);

    cv::Mat_<double> old_cam = cameraMatrix.clone(), old_dist = distCoeffs.clone();



    for (size_t ii = 0; ii < data.size(); ++ii) {
        local_rvecs[ii] = mat2vec(rvecs[ii]);
        local_tvecs[ii] = mat2vec(tvecs[ii]);

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

                >(new ProjectionFunctor(imagePoints[ii], objectPoints[ii]),
                  2*imagePoints[ii].size() // Number of residuals
                  );
        problem.AddResidualBlock(cost_function,
                                 nullptr, // Loss function (nullptr = L2)
                                 &cameraMatrix(0,0), // focal length x
                                 &cameraMatrix(1,1), // focal length y
                                 &cameraMatrix(0,2), // principal point x
                                 &cameraMatrix(1,2), // principal point y
                                 local_rvecs[ii].data(), // rotation vector for the target
                                 local_tvecs[ii].data(), // translation vector for the target
                                 local_dist.data() // distortion coefficients
                                 );

    }


    // Run the solver!
    ceres::Solver::Options options;
    options.num_threads = threads;
    options.max_num_iterations = 150;
    options.function_tolerance = 1e-16;
    options.gradient_tolerance = 1e-16;
    options.parameter_tolerance = 1e-16;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << "\n";
    std::cout << summary.FullReport() << "\n";

    size_t counter = 0;
    for (auto & it : distCoeffs) {
        it = local_dist[counter];
        counter++;
    }

    for (size_t ii = 0; ii < local_rvecs.size(); ++ii) {
        rvecs[ii] = vec2mat(local_rvecs[ii]);
        tvecs[ii] = vec2mat(local_tvecs[ii]);
    }

    if (verbose) {
        std::cout << "Parameters before: " << std::endl
                  << "Camera matrix: " << old_cam << std::endl
                  << "Distortion: " << old_dist << std::endl;
        std::cout << "Parameters after: " << std::endl
                  << "Camera matrix: " << cameraMatrix << std::endl
                  << "Distortion: " << distCoeffs << std::endl;
        std::cout << "Difference: old - new" << std::endl
                  << "Camera matrix: " << (old_cam - cameraMatrix) << std::endl
                  << "Distortion: " << (old_dist - distCoeffs) << std::endl;
    }

    return 0;
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

double Calib::CeresCalibFlexibleTarget() {
    std::cout << "Running Ceres calibration with flexible target." << std::endl;

    // Build the problem.
    ceres::Problem problem;

    prepareCalibration();

    std::vector<std::vector<double> > local_rvecs(data.size()), local_tvecs(data.size());

    std::vector<double> local_dist = mat2vec(distCoeffs);

    cv::Mat_<double> old_cam = cameraMatrix.clone(), old_dist = distCoeffs.clone();


    std::map<cv::Point3i, std::vector<double>, cmpSimpleIndex3<cv::Point3i> > local_corrections;

    for (const auto& it: data) {
        for (size_t ii = 0; ii < it.second.size(); ++ii) {
            cv::Point3i const c = getSimpleId(it.second.get(ii));
            local_corrections[c] = point2vec3f(objectPointCorrections[c]);
        }
    }

    std::set<cv::Point3i, cmpSimpleIndex3<cv::Point3i> > ids;

    for (size_t ii = 0; ii < data.size(); ++ii) {
        local_rvecs[ii] = mat2vec(rvecs[ii]);
        local_tvecs[ii] = mat2vec(tvecs[ii]);

        auto const & sub_data = data[imageFiles[ii]];
        for (size_t jj = 0; jj < sub_data.size(); ++jj) {
            cv::Point3i const c = getSimpleId(sub_data.get(jj));
            ids.insert(c);
            {
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

                        >(new FlexibleTargetProjectionFunctor(
                              imagePoints[ii][jj],
                              objectPoints[ii][jj]
                              )
                          );
                problem.AddResidualBlock(cost_function,
                                         new ceres::CauchyLoss(0.5), // Loss function (nullptr = L2)
                                         &cameraMatrix(0,0), // focal length x
                                         &cameraMatrix(1,1), // focal length y
                                         &cameraMatrix(0,2), // principal point x
                                         &cameraMatrix(1,2), // principal point y
                                         local_rvecs[ii].data(), // rotation vector for the target
                                         local_tvecs[ii].data(), // translation vector for the target
                                         local_corrections[c].data(),
                                         local_dist.data() // distortion coefficients
                                         );
            }
        }
    }

    for (cv::Point3i const& it : ids) {
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
    options.num_threads = threads;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.max_num_iterations = 150;
    options.minimizer_progress_to_stdout = true;
    options.function_tolerance = 1e-16;
    options.gradient_tolerance = 1e-16;
    options.parameter_tolerance = 1e-16;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << "\n";
    std::cout << summary.FullReport() << "\n";

    size_t counter = 0;
    for (auto & it : distCoeffs) {
        it = local_dist[counter];
        counter++;
    }

    for (auto const& it : local_corrections) {
        objectPointCorrections[it.first] = vec2point3f(it.second);
    }

    for (size_t ii = 0; ii < local_rvecs.size(); ++ii) {
        rvecs[ii] = vec2mat(local_rvecs[ii]);
        tvecs[ii] = vec2mat(local_tvecs[ii]);
    }

    if (verbose) {
        std::cout << "Parameters before: " << std::endl
                  << "Camera matrix: " << old_cam << std::endl
                  << "Distortion: " << old_dist << std::endl;
        std::cout << "Parameters after: " << std::endl
                  << "Camera matrix: " << cameraMatrix << std::endl
                  << "Distortion: " << distCoeffs << std::endl;
        std::cout << "Difference: old - new" << std::endl
                  << "Camera matrix: " << (old_cam - cameraMatrix) << std::endl
                  << "Distortion: " << (old_dist - distCoeffs) << std::endl;
    }

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
    if (0 == z) {
        z = 1;
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
    if (T(0) == z) {
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
    T c1 = T(1) - c;

    // Calculate normalized vector.
    T const factor = (T(0) != theta ? T(1)/theta : T(1));
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

ProjectionFunctor::ProjectionFunctor(
        const std::vector<Point2f> &_markers,
        const std::vector<Point3f> &_points) :
    markers(_markers),
    points(_points) {

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
    residuals[0] = result[0] - T(marker.x);
    residuals[1] = result[1] - T(marker.y);
    return true;
}


} // namespace hdcalib
