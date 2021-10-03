#undef NDEBUG;
#include <cassert>
#include "hdcalib.h"

namespace hdcalib {

template<int NUM, int DEG>
SplineRegularizer<NUM, DEG>::SplineRegularizer(const Size &_size) : size(_size) {}

template<int NUM, int DEG>
template<class T>
bool SplineRegularizer<NUM, DEG>::operator()(T const * const weights_x, T const * const weights_y, T * residuals) const {
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

template<class T>
T Calib:: evaluateSpline(T const x, int const POS, int const DEG) {
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
    return (x-pos)/deg*evaluateSpline(x,POS,DEG-1) + (T(POS+DEG+1)-x)/deg*evaluateSpline(x, POS+1, DEG-1);
}

template double Calib::evaluateSpline<double>(double const x, int const POS, int const DEG);
template float Calib::evaluateSpline<float>(float const x, int const POS, int const DEG);
template ceres::Jet<double, 1> Calib::evaluateSpline<ceres::Jet<double, 1> >(ceres::Jet<double, 1> const x, int const POS, int const DEG);
template ceres::Jet<double, 2> Calib::evaluateSpline<ceres::Jet<double, 2> >(ceres::Jet<double, 2> const x, int const POS, int const DEG);

template<class T>
T Calib::evaluateSplineDerivative(T const x, int const POS, int const DEG) {
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
    return hdcalib::Calib::evaluateSpline(x, POS, DEG-1) - hdcalib::Calib::evaluateSpline(x, POS+1, DEG-1);
}

template double Calib::evaluateSplineDerivative<double>(double const x, int const POS, int const DEG);
template float Calib::evaluateSplineDerivative<float>(float const x, int const POS, int const DEG);




template<int NUM, int DEG, class F, class T>
void Calib::projectSpline(
        F const p[3],
T result[2],
const T focal[2],
const T principal[2],
const T R[9],
const T t[3],
const T weights_x[(NUM+DEG)*(NUM+DEG)],
const T weights_y[(NUM+DEG)*(NUM+DEG)],
cv::Size const& size
) {
    T const X(p[0]);
    T const Y(p[1]);
    T const Z(p[2]);
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

    SplineFunctor<NUM, DEG>::apply(result, weights_x, weights_y, size);
}


template void Calib::projectSpline< 3, 3, double, double>(const double *, double *, const double *, const double *, const double *, const double *, const double *, const double *, cv::Size const&);
template void Calib::projectSpline< 5, 3, double, double>(const double *, double *, const double *, const double *, const double *, const double *, const double *, const double *, cv::Size const&);
template void Calib::projectSpline< 7, 3, double, double>(const double *, double *, const double *, const double *, const double *, const double *, const double *, const double *, cv::Size const&);
template void Calib::projectSpline< 9, 3, double, double>(const double *, double *, const double *, const double *, const double *, const double *, const double *, const double *, cv::Size const&);
template void Calib::projectSpline<11, 3, double, double>(const double *, double *, const double *, const double *, const double *, const double *, const double *, const double *, cv::Size const&);
template void Calib::projectSpline<13, 3, double, double>(const double *, double *, const double *, const double *, const double *, const double *, const double *, const double *, cv::Size const&);
template void Calib::projectSpline<15, 3, double, double>(const double *, double *, const double *, const double *, const double *, const double *, const double *, const double *, cv::Size const&);
template void Calib::projectSpline<17, 3, double, double>(const double *, double *, const double *, const double *, const double *, const double *, const double *, const double *, cv::Size const&);
template void Calib::projectSpline<19, 3, double, double>(const double *, double *, const double *, const double *, const double *, const double *, const double *, const double *, cv::Size const&);
template void Calib::projectSpline<21, 3, double, double>(const double *, double *, const double *, const double *, const double *, const double *, const double *, const double *, cv::Size const&);

template<int NUM, int DEG>
template<class T, class U>
void SplineFunctor<NUM, DEG>::apply(T* pt, U const * const weights_x, U const * const weights_y, cv::Size const& size) {
    T scaled_pt[2] = {pt[0]*T(double(NUM)/size.width), pt[1]*T(double(NUM)/size.height)};
    T const dx = applySingle(scaled_pt, weights_x);
    T const dy = applySingle(scaled_pt, weights_y);
    pt[0] += dx;
    pt[1] += dy;
}

template<int NUM, int DEG>
template<class T, class U>
T SplineFunctor<NUM, DEG>::apply_Dx(T const * const pt, U const * const weights, cv::Size const& size) {
    T scaled_pt[2] = {pt[0]*T(double(NUM)/size.width), pt[1]*T(double(NUM)/size.height)};
    return applySingleDx(scaled_pt, weights);
}

template<int NUM, int DEG>
template<class T, class U>
T SplineFunctor<NUM, DEG>::apply_Dy(T const * const pt, U const * const weights, cv::Size const& size) {
    T scaled_pt[2] = {pt[0]*T(double(NUM)/size.width), pt[1]*T(double(NUM)/size.height)};
    return applySingleDy(scaled_pt, weights);
}

template<int NUM, int DEG>
template<class T, class U>
T SplineFunctor<NUM, DEG>::applySingle(T const * const val, U const * const weights) {
    T col[SplineFunctor<NUM, DEG>::n_rows];
    for (size_t ii = 0; ii < SplineFunctor<NUM, DEG>::n_rows; ++ii) {
        col[ii] = applyRow(val[0], &weights[ii*SplineFunctor<NUM, DEG>::n_rows]);
    }
    return applyRow(val[1], col);
}

template<int NUM, int DEG>
template<class T, class U>
T SplineFunctor<NUM, DEG>::applySingleDx(T const * const val, U const * const weights) {
    T col[SplineFunctor<NUM, DEG>::n_rows];
    for (size_t ii = 0; ii < SplineFunctor<NUM, DEG>::n_rows; ++ii) {
        col[ii] = applyRowDerivative(val[0], &weights[ii*SplineFunctor<NUM, DEG>::n_rows]);
    }
    return applyRow(val[1], col);
}

template<int NUM, int DEG>
template<class T, class U>
T SplineFunctor<NUM, DEG>::applySingleDy(T const * const val, U const * const weights) {
    T col[SplineFunctor<NUM, DEG>::n_rows];
    for (size_t ii = 0; ii < SplineFunctor<NUM, DEG>::n_rows; ++ii) {
        col[ii] = applyRow(val[0], &weights[ii*SplineFunctor<NUM, DEG>::n_rows]);
    }
    return applyRowDerivative(val[1], col);
}

template<int NUM, int DEG>
template<class T, class U>
T SplineFunctor<NUM, DEG>::applyRow(T const& val, U const * const weights) {
    T result(0);
    for (int ii = 0; ii < int(SplineFunctor<NUM, DEG>::n_rows); ++ii) {
        int const POS = ii - DEG;
        result += weights[ii]*hdcalib::Calib::evaluateSpline(val, POS, DEG);
    }
    return result;
}

template<int NUM, int DEG>
template<class T, class U>
T SplineFunctor<NUM, DEG>::applyRowDerivative(T const& val, U const * const weights) {
    T result(0);
    for (int ii = 0; ii < int(SplineFunctor<NUM, DEG>::n_rows); ++ii) {
        int const POS = ii - DEG;
        result += weights[ii]*Calib::evaluateSplineDerivative(val, POS, DEG);
    }
    return result;
}

template<int NUM, int DEG>
class SplineProjectionFunctor {
    cv::Point2f const marker;
    cv::Point3f const point;
    cv::Size const size;
public:
    static const int n = (NUM+DEG)*(NUM+DEG);
    static const int n_rows = (NUM+DEG);
    double weight = 1;
    SplineProjectionFunctor(cv::Point2f const& _marker,
                            cv::Point3f const& _point,
                            cv::Size const& _size) : marker(_marker), point(_point), size(_size) {}
    /*
    1, // focal length x
    1, // focal length y
    1, // principal point x
    1, // principal point y
    3, // rotation vector for the target
    3, // translation vector for the target
    3, // correction vector for the 3d marker position
    (NUM+DEG)² // distortion coefficients for x
    (NUM+DEG)² // distortion coefficients for y
    */
    template<class T>
    bool operator()(
            T const* const f_x,
            T const* const f_y,
            T const* const c_x,
            T const* const c_y,
            T const* const rvec,
            T const* const tvec,
            T const* const correction,
            T const* const x_factor,
            T const* const weights_x,
            T const* const weights_y,
            T* residuals) const {
        T rot_mat[9];
        Calib::rot_vec2mat(rvec, rot_mat);
        T const f[2] = {f_x[0], f_y[0]};
        T const c[2] = {c_x[0], c_y[0]};
        T result[2] = {T(0), T(0)};
        T const src[3] = {T(point.x)*(T(1) + x_factor[0]/100.0) + correction[0],
                          T(point.y) + correction[1],
                          T(point.z) + correction[2]};
        Calib::projectSpline<NUM, DEG>(src, result, f, c, rot_mat, tvec, weights_x, weights_y, size);
        residuals[0] = T(weight) * (result[0] - T(marker.x));
        residuals[1] = T(weight) * (result[1] - T(marker.y));
        return true;
    }
};

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
void distortPt(
        F const p[2],
T result[2],
const T focal[2],
const T principal[2],
const T dist[14]
) {
    T& x = result[0];
    T& y = result[1];

    x = (p[0] - principal[0])/focal[0];
    y = (p[1] - principal[1])/focal[1];

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

template<int NUM, int DEG>
void Calib::transformModel2Spline(CalibResult& calib, std::vector<double>& weights_x, std::vector<double>& weights_y) {
    randutils::mt19937_rng rng;
    cv::Size const size = calib.calib->getImageSize();
    int const width = size.width;
    int const height = size.height;

    typedef SplineFunctor<NUM, DEG> Func;
    typedef SplineRegularizer<NUM, DEG> Reg;

    double const focal[2] = {calib.cameraMatrix(0,0), calib.cameraMatrix(1,1)};
    double const principal[2] = {calib.cameraMatrix(0,2), calib.cameraMatrix(1,2)};
    std::vector<double> dist = mat2vec(calib.distCoeffs);
    std::vector<cv::Vec2f> distorted, undistorted;

    int const num_val = 30;
    for (int xx = 0; xx <= num_val; ++xx) {
        double const x = xx*width/num_val;
        for (int yy = 0; yy <= num_val; ++yy) {
            double const y = yy*height/num_val;
            double p[2] = {x,y};
            cv::Vec2d _distorted(0,0);
            distortPt(p, _distorted.val, focal, principal, dist.data());

            undistorted.push_back(cv::Vec2f(x, y));
            distorted.push_back(_distorted);
        }
    }
    assert(distorted.size() == undistorted.size());

    weights_x = std::vector<double>(Func::n, 0.0);
    weights_y = std::vector<double>(Func::n, 0.0);

    ceres::Problem problem;
    runningstats::QuantileStats<float> pre_fit_undistorted_distances;
    runningstats::QuantileStats<float> pre_fit_distorted_distances;
    for (size_t ii = 0; ii < distorted.size(); ++ii) {
        if (undistorted[ii][0] < 0 || undistorted[ii][1] < 0 || undistorted[ii][0] > width || undistorted[ii][1] > height) {
            continue;
        }
        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<Func, 2, Func::n, Func::n>(
                                     new Func(undistorted[ii], distorted[ii], size)),
                                 nullptr,
                                 weights_x.data(),
                                 weights_y.data()
                                 );
        cv::Vec2f result = undistorted[ii];
        Func(undistorted[ii], distorted[ii], size).apply(result.val, weights_x.data(), weights_y.data());
        pre_fit_undistorted_distances.push_unsafe(cv::norm(result - undistorted[ii]));
        pre_fit_distorted_distances.push_unsafe(cv::norm(result - distorted[ii]));
    }
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<Reg, Reg::n, Func::n, Func::n>(
                                 new Reg(getImageSize())),
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

    runningstats::QuantileStats<float> post_fit_undistorted_distances;
    runningstats::QuantileStats<float> post_fit_distorted_distances;
    for (size_t ii = 0; ii < distorted.size(); ++ii) {
        if (undistorted[ii][0] < 0 || undistorted[ii][1] < 0 || undistorted[ii][0] > width || undistorted[ii][1] > height) {
            continue;
        }
        cv::Vec2f result = undistorted[ii];
        Func(undistorted[ii], distorted[ii], size).apply(result.val, weights_x.data(), weights_y.data());
        post_fit_undistorted_distances.push_unsafe(cv::norm(result - undistorted[ii]));
        post_fit_distorted_distances.push_unsafe(cv::norm(result - distorted[ii]));
    }

    clog::L("Calib::transformModel2Spline", 2) << "pre-fit undistorted: " << pre_fit_undistorted_distances.print();
    clog::L("Calib::transformModel2Spline", 2) << "pre-fit distorted: " << pre_fit_distorted_distances.print();

    clog::L("Calib::transformModel2Spline", 2) << "post-fit undistorted: " << post_fit_undistorted_distances.print();
    clog::L("Calib::transformModel2Spline", 2) << "post-fit distorted: " << post_fit_distorted_distances.print();

}

void distortSpline(
        cv::Vec2f & pt,
        int const NUM,
        int const DEG,
        std::vector<double> const& weights_x,
        std::vector<double> const& weights_y,
        cv::Size const& size) {
    assert (3 == DEG);
    switch(NUM) {
    case 3:
        SplineFunctor<3, 3>::apply(pt.val, weights_x.data(), weights_y.data(), size); return;
    case 5:
        SplineFunctor<5, 3>::apply(pt.val, weights_x.data(), weights_y.data(), size); return;
    case 7:
        SplineFunctor<7, 3>::apply(pt.val, weights_x.data(), weights_y.data(), size); return;
    case 9:
        SplineFunctor<9, 3>::apply(pt.val, weights_x.data(), weights_y.data(), size); return;
    default:
        throw std::runtime_error("distortSpline called with invalid parameters");
    }
}

template<int NUM, int DEG>
void Calib::transformSpline2Spline(CalibResult& calib, std::vector<double>& weights_x, std::vector<double>& weights_y) {
    randutils::mt19937_rng rng;
    cv::Size const size = calib.calib->getImageSize();
    int const width = size.width;
    int const height = size.height;

    typedef SplineFunctor<NUM, DEG> F;

    std::vector<double> src_x = calib.spline_x;
    std::vector<double> src_y = calib.spline_y;

    assert(src_x.size() == src_y.size());
    int const N_old = std::round(std::sqrt(src_x.size())) - DEG;
    assert(size_t((N_old + DEG) * (N_old + DEG)) == src_x.size());
    clog::L("Calib::transformSpline2Spline", 2) << "Spline " << N_old << " -> " << NUM << std::endl;

    std::vector<double> dist = mat2vec(calib.distCoeffs);
    std::vector<cv::Vec2f> distorted, undistorted;
    for (size_t ii = 0; ii < 10'000; ++ii) {
        double const x = rng.uniform<double>(0, width);
        double const y = rng.uniform<double>(0, height);
        cv::Vec2f _distorted(x, y);
        distortSpline(_distorted, N_old, DEG, src_x, src_y, size);

        undistorted.push_back(cv::Vec2f(x, y));
        distorted.push_back(_distorted);
    }
    assert(distorted.size() == undistorted.size());

    weights_x = std::vector<double>(F::n, 0.0);
    weights_y = std::vector<double>(F::n, 0.0);

    ceres::Problem problem;
    for (size_t ii = 0; ii < distorted.size(); ++ii) {
        if (undistorted[ii][0] < 0 || undistorted[ii][1] < 0 || undistorted[ii][0] > width || undistorted[ii][1] > height) {
            continue;
        }
        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<F, 2, F::n, F::n>(
                                     new F(undistorted[ii], distorted[ii], size)),
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
}

template<int NUM, int DEG>
double Calib::CeresCalibFlexibleTargetSplineSub(double const outlier_threshold, FixedValues fixed) {
    prepareCalibration();

    std::string const name = std::string("Spline-") + std::to_string(NUM) + "-" + std::to_string(DEG);
    ceres::Problem problem;

    CalibResult & calib = calibrations[name];
    calib.calib = this;
    calib.name = name;

    typedef SplineProjectionFunctor<NUM, DEG> Func;
    typedef SplineRegularizer<NUM, DEG> Reg;

    // Initialize spline parameter vectors if necessary or scale version with lower resolution if available.
    scaleSquareMatVec(calib.spline_x, Func::n_rows);
    scaleSquareMatVec(calib.spline_y, Func::n_rows);

    std::vector<std::vector<double> > local_rvecs(data.size()), local_tvecs(data.size());

    cv::Mat_<double> old_cam = calib.cameraMatrix.clone();
    std::vector<double> old_dist = calib.distN;


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

        auto const & sub_data = imagePoints[ii];
        size_t outlier_counter = 0;
        runningstats::QuantileStats<float> error_stats;
        for (size_t jj = 0; jj < sub_data.size(); ++jj) {
            cv::Scalar_<int> const c = this->ids[ii][jj];
            ids.insert(c);
            {
                Func loss (
                            imagePoints[ii][jj],
                            objectPoints[ii][jj],
                            imageSize
                            );
                double residuals[2] = {0,0};
                if (outlier_threshold > 0) {
                    loss(&calib.cameraMatrix(0,0), // focal length x
                         &calib.cameraMatrix(1,1), // focal length y
                         &calib.cameraMatrix(0,2), // principal point x
                         &calib.cameraMatrix(1,2), // principal point y
                         local_rvecs[ii].data(), // rotation vector for the target
                         local_tvecs[ii].data(), // translation vector for the target
                         local_corrections[c].data(), // 3d point correction
                         &calib.x_factor,
                         calib.spline_x.data(),
                         calib.spline_y.data(),
                         residuals);
                    double const error = std::sqrt(residuals[0] * residuals[0] + residuals[1] * residuals[1]);
                    if (error > outlier_threshold) {
                        outlier_counter++;
                        continue;
                    }
                    else {
                        error_stats.push_unsafe(error);
                    }
                }
                if (ignore_current_file) {
                    continue;
                }
                ceres::CostFunction * cost_function =
                        new ceres::AutoDiffCostFunction<
                        Func,
                        2, // Number of residuals
                        1, // focal length x
                        1, // focal length y
                        1, // principal point x
                        1, // principal point y
                        3, // rotation vector for the target
                        3, // translation vector for the target
                        3, // correction vector for the 3d marker position
                        1, // x-factor
                        Func::n, // spline distortion coefficients x
                        Func::n // spline distortion coefficients y

                        >(new Func (
                              imagePoints[ii][jj],
                              objectPoints[ii][jj],
                              imageSize
                              ));
                problem.AddResidualBlock(cost_function,
                                         cauchy_param > 0 ? new ceres::CauchyLoss(cauchy_param) : nullptr, // Loss function (nullptr = L2)
                                         &calib.cameraMatrix(0,0), // focal length x
                                         &calib.cameraMatrix(1,1), // focal length y
                                         &calib.cameraMatrix(0,2), // principal point x
                                         &calib.cameraMatrix(1,2), // principal point y
                                         local_rvecs[ii].data(), // rotation vector for the target
                                         local_tvecs[ii].data(), // translation vector for the target
                                         local_corrections[c].data(), // 3d point correction
                                         &calib.x_factor,
                                         calib.spline_x.data(), // distortion coefficients
                                         calib.spline_y.data() // distortion coefficients
                                         );
            }
        }
        if (fixed.rvecs) {
            problem.SetParameterBlockConstant(local_rvecs[ii].data());
        }
        if (fixed.tvecs) {
            problem.SetParameterBlockConstant(local_tvecs[ii].data());
        }
        double const outlier_percentage = 100.0 * double(outlier_counter) / sub_data.size();
        outlier_percentages.push_unsafe(outlier_percentage);
        outlier_ranking.insert({outlier_percentage, imageFiles[ii] + " median: " + std::to_string(error_stats.getMedian())});
        calib.outlier_percentages[ii] = outlier_percentage;
    }
    if (fixed.focal) {
        problem.SetParameterBlockConstant(&calib.cameraMatrix(0,0));
        problem.SetParameterBlockConstant(&calib.cameraMatrix(1,1));
    }
    if (fixed.principal) {
        problem.SetParameterBlockConstant(&calib.cameraMatrix(0,2));
        problem.SetParameterBlockConstant(&calib.cameraMatrix(1,2));
    }
    if (fixed.x_factor) {
        problem.SetParameterBlockConstant(&calib.x_factor);
    }

    clog::L(name, 2) << "Outlier ranking:" << std::endl;
    for (auto const& it : outlier_ranking) {
        clog::L(name, 2) << it.second << ": \t" << it.first << std::endl;
    }
    std::cout << "Ignored " << 100.0 * double(ignored_files_counter) / data.size() << "% of files" << std::endl;
    if (outlier_threshold > 0) {
        clog::L(name, 2) << "Outlier percentage stats: " << outlier_percentages.print() << std::endl;
    }

    for (cv::Scalar_<int> const& it : ids) {
        if (fixed.objCorr) {
            problem.SetParameterBlockConstant(local_corrections[it].data());
        }
        else {
            ceres::CostFunction* cost_function =
                    new ceres::AutoDiffCostFunction<LocalCorrectionsSum<3>, 3, 3>(
                        new LocalCorrectionsSum<3>(1));
            problem.AddResidualBlock(cost_function,
                                     nullptr, // Loss function (nullptr = L2)
                                     local_corrections[it].data() // correction
                                     );
        }
    }
    if (fixed.dist) {
        problem.SetParameterBlockConstant(calib.spline_x.data());
        problem.SetParameterBlockConstant(calib.spline_y.data());
    }
    else {
        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<Reg, Reg::n, Func::n, Func::n>(
                                     new Reg(getImageSize())),
                                 nullptr,
                                 calib.spline_x.data(),
                                 calib.spline_y.data()
                                 );
    }
    // Run the solver!
    ceres::Solver::Options options;
    options.num_threads = int(threads);
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.max_num_iterations = max_iter;
    options.minimizer_progress_to_stdout = verbose;
    options.function_tolerance = ceres_tolerance;
    options.gradient_tolerance = ceres_tolerance;
    options.parameter_tolerance = ceres_tolerance;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    clog::L(name, 1) << summary.BriefReport() << "\n";
    clog::L(name, 1) << summary.FullReport() << "\n";

    for (auto const& it : local_corrections) {
        calib.objectPointCorrections[it.first] = vec2point3f(it.second);
    }

    for (size_t ii = 0; ii < local_rvecs.size(); ++ii) {
        calib.rvecs[ii] = vec2mat(local_rvecs[ii]);
        calib.tvecs[ii] = vec2mat(local_tvecs[ii]);
    }

    std::vector<Corner> all_corners;
    std::vector<cv::Vec2f> all_residuals;
    runningstats::QuantileStats<float> error_stats;
    for (size_t ii = 0; ii < data.size(); ++ii) {
        auto const & sub_data = imagePoints[ii];
        size_t outlier_counter = 0;
        for (size_t jj = 0; jj < sub_data.size(); ++jj) {
            cv::Scalar_<int> const c = this->ids[ii][jj];
            {
                Func loss (
                            imagePoints[ii][jj],
                            objectPoints[ii][jj],
                            imageSize
                            );
                cv::Vec2d residuals;
                loss(&calib.cameraMatrix(0,0), // focal length x
                     &calib.cameraMatrix(1,1), // focal length y
                     &calib.cameraMatrix(0,2), // principal point x
                     &calib.cameraMatrix(1,2), // principal point y
                     local_rvecs[ii].data(), // rotation vector for the target
                     local_tvecs[ii].data(), // translation vector for the target
                     local_corrections[c].data(), // 3d point correction
                     &calib.x_factor,
                     calib.spline_x.data(),
                     calib.spline_y.data(),
                     residuals.val);
                double const error = cv::norm(residuals);
                error_stats.push_unsafe(error);
                all_residuals.push_back(residuals);
                all_corners.push_back(corners[ii][jj]);
            }
        }
        double const outlier_percentage = 100.0 * double(outlier_counter) / sub_data.size();
        outlier_percentages.push_unsafe(outlier_percentage);
        outlier_ranking.insert({outlier_percentage, imageFiles[ii] + " median: " + std::to_string(error_stats.getMedian())});
        calib.outlier_percentages[ii] = outlier_percentage;
    }
    saveReprojectionsJsons(cache_prefix + "-" + calib.name + "-reprojection-errors", all_corners, all_residuals);
    clog::L(name, 1) << "Error stats: " << error_stats.print();
    clog::L(name, 1) << "Median error: " << error_stats.getMedian() << ", IQR: [" << error_stats.getQuantile(.25) << ", " << error_stats.getQuantile(.75) << "]";
    clog::L(name, 1) << "Parameters before: " << std::endl
                     << "Camera matrix: " << old_cam << std::endl
                        ;//<< "Distortion: " << old_dist << std::endl;
    clog::L(name, 1) << "Parameters after: " << std::endl
                     << "Camera matrix: " << calib.cameraMatrix << std::endl
                        ;//<< "Distortion: " << calib.distCoeffs << std::endl;
    clog::L(name, 1) << "Difference: old - new" << std::endl
                     << "Camera matrix: " << (old_cam - calib.cameraMatrix) << std::endl
                        ;//<< "Distortion: " << (old_dist - calib.distCoeffs) << std::endl;
    clog::L(name, 1) << "x_factor: " << calib.x_factor;
    //CeresCalibKnownCorrections(outlier_threshold, calib);
    return 0;

}

template<int NUM, int DEG>
double Calib::CeresCalibFlexibleTargetSpline(double const outlier_threshold) {
    // Build the problem.

    prepareCalibration();

    std::string const name = std::string("Spline-") + std::to_string(NUM) + "-" + std::to_string(DEG);
    if (!hasCalibName(name)) {
        /*
        for (int N = NUM-2; N >= 3; N -= 2) {
            std::string const n = std::string("Spline-") + std::to_string(N) + "-" + std::to_string(DEG);
            if (!hasCalibName(name) && hasCalibName(n)) {
                calibrations[name] = calibrations[n];
                //transformSpline2Spline<NUM, DEG>(calibrations[name], calibrations[name].spline_x, calibrations[name].spline_y);
                break;
            }
        }
        //*/
        for (std::string const& n : {"Flexible", "SemiFlexible", "Ceres", "OpenCV", "SimpleCeres", "SimpleOpenCV"}) {
            if (!hasCalibName(name) && hasCalibName(n)) {
                calibrations[name] = calibrations[n];
                //transformModel2Spline<NUM, DEG>(calibrations[name], calibrations[name].spline_x, calibrations[name].spline_y);
                break;
            }
        }
    }

    FixedValues f1(true);
    f1.dist = false;
    CeresCalibFlexibleTargetSplineSub<NUM, DEG>(outlier_threshold, f1);

    FixedValues f2(false);
    return CeresCalibFlexibleTargetSplineSub<NUM, DEG>(outlier_threshold, f2);

}

template double Calib::CeresCalibFlexibleTargetSpline< 3,3>(double const outlier_threshold);
template double Calib::CeresCalibFlexibleTargetSpline< 5,3>(double const outlier_threshold);
template double Calib::CeresCalibFlexibleTargetSpline< 7,3>(double const outlier_threshold);
template double Calib::CeresCalibFlexibleTargetSpline< 9,3>(double const outlier_threshold);
template double Calib::CeresCalibFlexibleTargetSpline<11,3>(double const outlier_threshold);
template double Calib::CeresCalibFlexibleTargetSpline<13,3>(double const outlier_threshold);
template double Calib::CeresCalibFlexibleTargetSpline<15,3>(double const outlier_threshold);
template double Calib::CeresCalibFlexibleTargetSpline<17,3>(double const outlier_threshold);
template double Calib::CeresCalibFlexibleTargetSpline<19,3>(double const outlier_threshold);
template double Calib::CeresCalibFlexibleTargetSpline<21,3>(double const outlier_threshold);


template<int NUM, int DEG>
SplineFunctor<NUM, DEG>::SplineFunctor(const Vec2f &_src, const Vec2f &_dst, const Size &_size) :
    src(_src),
    dst(_dst),
    size(_size),
    factor_x(double(NUM)/size.width), factor_y(double(NUM)/size.height){}

template<int NUM, int DEG>
Vec2f SplineFunctor<NUM, DEG>::apply(const Vec2f &pt, const cv::Mat_<float> &weights_x, cv::Mat_<float> const& weights_y) const {
    cv::Vec2f result(pt);
    apply(result.val, weights_x.ptr<float>(), weights_y.ptr<float>());
    return result;
}

template struct SplineFunctor<3, 3>;
template struct SplineFunctor<5, 3>;
template struct SplineFunctor<7, 3>;
template struct SplineFunctor<9, 3>;

template bool SplineFunctor<3, 3>::operator()<ceres::Jet<double, 72> >(
        ceres::Jet<double, 72> const * const,
        ceres::Jet<double, 72> const * const,
        ceres::Jet<double, 72> *) const;

template bool SplineFunctor<5, 3>::operator()<ceres::Jet<double, 128> >(
        ceres::Jet<double, 128> const * const,
        ceres::Jet<double, 128> const * const,
        ceres::Jet<double, 128> *) const;

template bool SplineFunctor<7, 3>::operator()<ceres::Jet<double, 200> >(
        ceres::Jet<double, 200> const * const,
        ceres::Jet<double, 200> const * const,
        ceres::Jet<double, 200> *) const;

template bool SplineFunctor<9, 3>::operator()<ceres::Jet<double, 288> >(
        ceres::Jet<double, 288> const * const,
        ceres::Jet<double, 288> const * const,
        ceres::Jet<double, 288> *) const;


template bool SplineFunctor<3, 3>::operator()<double>(
        double const * const,
        double const * const,
        double *) const;

template bool SplineFunctor<5, 3>::operator()<double>(
        double const * const,
        double const * const,
        double *) const;

template bool SplineFunctor<7, 3>::operator()<double>(
        double const * const,
        double const * const,
        double *) const;

template bool SplineFunctor<9, 3>::operator()<double>(
        double const * const,
        double const * const,
        double *) const;


template<int NUM, int DEG>
Vec2f SplineFunctor<NUM, DEG>::apply(const Vec2f &pt, const std::vector<double> &weights_x, std::vector<double> weights_y) const {
    cv::Vec2f result(pt);
    apply(result.val, weights_x.data(), weights_y.data());
    return result;
}

template<int NUM, int DEG>
template<class T>
bool SplineFunctor<NUM, DEG>::operator()(T const * const weights_x, T const * const weights_y, T * residuals) const {
    residuals[0] = T(src[0]);
    residuals[1] = T(src[1]);
    apply(residuals, weights_x, weights_y);
    residuals[0] -= T(dst[0]);
    residuals[1] -= T(dst[1]);
    return true;
}

template<int NUM, int DEG>
template<class T, class U>
void SplineFunctor<NUM, DEG>::apply(T* pt, U const * const weights_x, U const * const weights_y) const {
    T scaled_pt[2] = {pt[0]*T(factor_x), pt[1]*T(factor_y)};
    T const dx = applySingle(scaled_pt, weights_x);
    T const dy = applySingle(scaled_pt, weights_y);
    pt[0] += dx;
    pt[1] += dy;
}

} // namespace hdcalib
