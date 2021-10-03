#include "hdcalib.h"
#include "libplotoptflow.h"
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
    if (preparedCalib && imagePoints.size() == data.size() && objectPoints.size() == data.size() && preparedMarkerCountLimit == markerCountLimit) {
        return;
    }
    preparedOpenCVCalib = false;
    preparedOpenCVROCalib = false;
    preparedCalib = true;
    preparedMarkerCountLimit = markerCountLimit;

    if (markerCountLimit > 0) {
        markerSelection.clear();
        for (auto& it : data) {
            for (size_t ii = 0; ii < it.second.size(); ++ii) {
                markerSelection.insert(it.second.get(ii).getSimpleIdLayer());
            }
        }
        std::vector<cv::Scalar_<int> > tmp(markerSelection.begin(), markerSelection.end());
        if (int(tmp.size()) > markerCountLimit) {
            randutils::mt19937_rng rng;
            rng.sample(markerCountLimit, tmp);
            tmp.resize(markerCountLimit);
            markerSelection.clear();
            markerSelection.insert(tmp.begin(), tmp.end());
        }
    }

    corners = std::vector<std::vector<Corner> >(data.size());
    imagePoints = std::vector<std::vector<cv::Point2f> >(data.size());
    objectPoints = std::vector<std::vector<cv::Point3f> >(data.size());
    ids = std::vector<std::vector<cv::Scalar_<int> > >(data.size());
    imageFiles.resize(data.size());

    runningstats::RunningStats points_stats;
    size_t ii = 0;
    for (std::pair<const std::string, CornerStore> const& it : data) {
        //it.second.getPointsMainMarkerAdjacent(imagePoints[ii], objectPoints[ii], ids[ii], *this);
        it.second.getPoints(imagePoints[ii], objectPoints[ii], ids[ii], corners[ii], *this);
        imageFiles[ii] = it.first;
        points_stats.push_unsafe(imagePoints[ii].size());
        ++ii;
    }
    clog::L(__FUNCTION__, 2) << "#points stats: " << points_stats.print();
}

void CornerStore::getPoints(
        std::vector<Point2f> &imagePoints,
        std::vector<Point3f> &objectPoints,
        std::vector<cv::Scalar_<int> > &ids,
        std::vector<Corner> &corners,
        hdcalib::Calib const& calib) const {
    imagePoints.clear();
    objectPoints.clear();
    ids.clear();
    corners.clear();
    int const factor_x = calib.getRestrictFactorX();
    int const mod_x = calib.getRestrictModX();

    int const factor_y = calib.getRestrictFactorY();
    int const mod_y = calib.getRestrictModY();

    for (size_t ii = 0; ii < size(); ++ii) {
        hdmarker::Corner const& c = get(ii);
        if (calib.getMarkerCountLimit() > 0) {
            if (!calib.isSelected(c)) {
                continue;
            }
        }
        if (factor_x > 0 && mod_x >= 0 && mod_x < factor_x && mod_x != (c.id.x % factor_x)) {
            continue;
        }
        if (factor_y > 0 && mod_y >= 0 && mod_y < factor_y && mod_y != (c.id.y % factor_y)) {
            continue;
        }
        corners.push_back(c);
        imagePoints.push_back(c.p);
        objectPoints.push_back(calib.getInitial3DCoord(c));
        ids.push_back(c.getSimpleIdLayer());
    }
}

void CornerStore::getPointsMainMarkerAdjacent(
        std::vector<Point2f> &imagePoints,
        std::vector<Point3f> &objectPoints,
        std::vector<cv::Scalar_<int> > &ids,
        hdcalib::Calib const& calib) const {
    imagePoints.clear();
    objectPoints.clear();
    ids.clear();
    std::vector<Corner> const pts = getMainMarkerAdjacent();
    for (size_t ii = 0; ii < pts.size(); ++ii) {
        hdmarker::Corner const& c = pts[ii];
        imagePoints.push_back(c.p);
        objectPoints.push_back(calib.getInitial3DCoord(c));
        ids.push_back(c.getSimpleIdLayer());
    }
}

void CornerStore::getPointsMinLayer(
        std::vector<Point2f> &imagePoints,
        std::vector<Point3f> &objectPoints,
        std::vector<cv::Scalar_<int> > &ids,
        hdcalib::Calib const& calib,
        int const min_layer) const {
    imagePoints.clear();
    objectPoints.clear();
    ids.clear();
    for (size_t ii = 0; ii < size(); ++ii) {
        hdmarker::Corner const& c = get(ii);
        if (calib.getMarkerCountLimit() > 0) {
            if (!calib.isSelected(c)) {
                continue;
            }
        }
        if (c.layer >= min_layer) {
            imagePoints.push_back(c.p);
            objectPoints.push_back(calib.getInitial3DCoord(c));
            ids.push_back(c.getSimpleIdLayer());
        }
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
    calib.calib = this;
    calib.name = "Ceres";

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

        std::vector<cv::Point2d> reprojections;
        std::vector<Corner> markers;
        calib.getReprojections(image_index, markers, reprojections);
        std::vector<cv::Point2f> local_image_points;
        std::vector<cv::Point3f> local_object_points;
        if (outlier_threshold < 0) {
            local_image_points = imagePoints[image_index];
            local_object_points = objectPoints[image_index];
        }
        else {
            for (size_t ii = 0; ii < markers.size() && ii < reprojections.size(); ++ii) {
                cv::Point2d const& marker = markers[ii].p;
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
    options.max_num_iterations = max_iter;
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

    runningstats::QuantileStats<float> error_stats;

    std::vector<cv::Vec2f> all_residuals;
    std::vector<Corner> all_features;
    for (size_t image_index = 0; image_index < data.size(); ++image_index) {
        local_rvecs[image_index] = mat2vec(calib.rvecs[image_index]);
        local_tvecs[image_index] = mat2vec(calib.tvecs[image_index]);

        std::vector<cv::Point2d> reprojections;
        std::vector<Corner> markers;
        calib.getReprojections(image_index, markers, reprojections);
        std::vector<cv::Point2f> local_image_points;
        std::vector<cv::Point3f> local_object_points;
        for (size_t ii = 0; ii < markers.size() && ii < reprojections.size(); ++ii) {
            Corner const& marker = markers[ii];
            cv::Point2d const& reprojection = reprojections[ii];
            double const error = distance<cv::Point2d>(marker.p, reprojection);
            error_stats.push_unsafe(error);
            all_features.push_back(marker);
            all_residuals.push_back(cv::Vec2f(reprojection.x - marker.p.x, reprojection.y - marker.p.y));
        }
    }
    saveReprojectionsJsons(cache_prefix + "-" + calib.name + "-reprojection-errors", all_features, all_residuals);

    calib.error_percentiles.clear();
    for (int ii = 0; ii <= 100; ++ii) {
        calib.error_percentiles.push_back(error_stats.getQuantile(double(ii/100)));
    }
    calib.error_median = error_stats.getMedian();

    clog::L(__func__, 1) << "Median error: " << calib.error_median << ", IQR: [" << error_stats.getQuantile(.25) << ", " << error_stats.getQuantile(.75) << "]";
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

double Calib::CeresCalibKnownCorrections(double const outlier_threshold,
                                         CalibResult & calib) {
    prepareCalibration();

    ceres::Problem problem;
    std::vector<std::vector<double> > local_rvecs(data.size()), local_tvecs(data.size());

    std::vector<double> local_dist = mat2vec(calib.distCoeffs);

    cv::Mat_<double> old_cam = calib.cameraMatrix.clone();
    cv::Mat_<double> old_dist = calib.distCoeffs.clone();

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
        runningstats::QuantileStats<float> error_stats;
        for (size_t jj = 0; jj < imagePoints[ii].size(); ++jj) {
            cv::Scalar_<int> const c = this->ids[ii][jj];
            {
                SingleProjectionFunctor loss (
                            imagePoints[ii][jj],
                            objectPoints[ii][jj] + calib.objectPointCorrections[c]
                            );
                double residuals[2] = {0,0};
                if (outlier_threshold > 0) {
                    loss(&calib.cameraMatrix(0,0), // focal length x
                         &calib.cameraMatrix(1,1), // focal length y
                         &calib.cameraMatrix(0,2), // principal point x
                         &calib.cameraMatrix(1,2), // principal point y
                         local_rvecs[ii].data(), // rotation vector for the target
                         local_tvecs[ii].data(), // translation vector for the target
                         local_dist.data(),
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
                        SingleProjectionFunctor,
                        2, // Number of residuals
                        1, // focal length x
                        1, // focal length y
                        1, // principal point x
                        1, // principal point y
                        3, // rotation vector for the target
                        3, // translation vector for the target
                        14 // distortion coefficients

                        >(new SingleProjectionFunctor (
                              imagePoints[ii][jj],
                              objectPoints[ii][jj] + calib.objectPointCorrections[c]
                              ));
                problem.AddResidualBlock(cost_function,
                                         cauchy_param > 0 ? new ceres::CauchyLoss(cauchy_param) : nullptr, // Loss function (nullptr = L2)
                                         &calib.cameraMatrix(0,0), // focal length x
                                         &calib.cameraMatrix(1,1), // focal length y
                                         &calib.cameraMatrix(0,2), // principal point x
                                         &calib.cameraMatrix(1,2), // principal point y
                                         local_rvecs[ii].data(), // rotation vector for the target
                                         local_tvecs[ii].data(), // translation vector for the target
                                         local_dist.data() // distortion coefficients
                                         );
            }
        }
        double const outlier_percentage = 100.0 * double(outlier_counter) / sub_data.size();
        outlier_percentages.push_unsafe(outlier_percentage);
        outlier_ranking.insert({outlier_percentage, imageFiles[ii] + " median: " + std::to_string(error_stats.getMedian())});
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

    clog::L(__func__, 1) << summary.BriefReport() << "\n";
    clog::L(__func__, 1) << summary.FullReport() << "\n";

    size_t counter = 0;
    for (auto & it : calib.distCoeffs) {
        it = local_dist[counter];
        counter++;
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
    calib.calib = this;
    calib.name = "SimpleCeres";

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
    options.max_num_iterations = max_iter;
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
    calib.calib = this;
    calib.name = "Flexible";

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
    runningstats::QuantileStats<float> markers_per_image;
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
        runningstats::QuantileStats<float> error_stats;
        size_t marker_count = 0;
        for (size_t jj = 0; jj < imagePoints[ii].size(); ++jj) {
            cv::Scalar_<int> const c = this->ids[ii][jj];
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
                         &calib.x_factor,
                         local_dist.data(),
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
                        FlexibleTargetProjectionFunctor,
                        2, // Number of residuals
                        1, // focal length x
                        1, // focal length y
                        1, // principal point x
                        1, // principal point y
                        3, // rotation vector for the target
                        3, // translation vector for the target
                        3, // correction vector for the 3d marker position
                        1, // x_factor
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
                                         &calib.x_factor,
                                         local_dist.data() // distortion coefficients
                                         );
                marker_count++;
            }
        }
        markers_per_image.push_unsafe(marker_count);
        double const outlier_percentage = 100.0 * double(outlier_counter) / sub_data.size();
        outlier_percentages.push_unsafe(outlier_percentage);
        outlier_ranking.insert({outlier_percentage, imageFiles[ii] + " median: " + std::to_string(error_stats.getMedian())});
        calib.outlier_percentages[ii] = outlier_percentage;
    }
    clog::L(__func__, 2) << "Markers per image: " << markers_per_image.print();
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
                new ceres::AutoDiffCostFunction<LocalCorrectionsSum<3>, 3, 3>(
                    new LocalCorrectionsSum<3>(1));
        problem.AddResidualBlock(cost_function,
                                 nullptr, // Loss function (nullptr = L2)
                                 local_corrections[it].data() // correction
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

    clog::L(__func__, 1) << summary.BriefReport() << "\n";
    clog::L(__func__, 1) << summary.FullReport() << "\n";

    std::vector<cv::Vec2f> all_residuals;
    std::vector<Corner> all_features;
    size_t downscale_factor = 50;
    cv::Size downscaled_size(imageSize.width/downscale_factor+1, imageSize.height/downscale_factor+1);
    cv::Mat_<cv::Vec2f> residuals_image(downscaled_size, cv::Vec2f(0,0));
    cv::Mat_<int> residuals_count(downscaled_size, int(0));
    runningstats::QuantileStats<float> error_stats;
    for (size_t ii = 0; ii < data.size(); ++ii) {
        for (size_t jj = 0; jj < imagePoints[ii].size(); ++jj) {
            cv::Scalar_<int> const c = this->ids[ii][jj];
            {
                FlexibleTargetProjectionFunctor loss (
                            imagePoints[ii][jj],
                            objectPoints[ii][jj]
                            );
                cv::Vec2d residuals(0,0);
                loss(&calib.cameraMatrix(0,0), // focal length x
                     &calib.cameraMatrix(1,1), // focal length y
                     &calib.cameraMatrix(0,2), // principal point x
                     &calib.cameraMatrix(1,2), // principal point y
                     local_rvecs[ii].data(), // rotation vector for the target
                     local_tvecs[ii].data(), // translation vector for the target
                     local_corrections[c].data(),
                     &calib.x_factor,
                     local_dist.data(),
                     residuals.val);
                double const error = cv::norm(residuals);
                error_stats.push_unsafe(cv::norm(residuals));
                cv::Point2i pt(std::round(imagePoints[ii][jj].x/downscale_factor), std::round(imagePoints[ii][jj].y/downscale_factor));
                residuals_image(pt) += residuals;
                residuals_count(pt)++;
                all_features.push_back(corners[ii][jj]);
                all_residuals.push_back(residuals);            }
        }
    }
    for (int ii = 0; ii < residuals_image.rows; ++ii) {
        for (int jj = 0; jj < residuals_image.cols; ++jj) {
            if (residuals_count(ii, jj) > 0) {
                residuals_image(ii, jj) /= residuals_count(ii, jj);
            }
            else {
                residuals_image(ii, jj)[0] = residuals_image(ii, jj)[1] = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
    saveReprojectionsJsons(cache_prefix + "-" + calib.name + "-reprojection-errors", all_features, all_residuals);
    hdflow::gnuplotWithArrows("FlexibleAfterCalib-residuals", residuals_image, -1, 100);

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

    calib.error_percentiles.clear();
    for (int ii = 0; ii <= 100; ++ii) {
        calib.error_percentiles.push_back(error_stats.getQuantile(double(ii/100)));
    }
    calib.error_median = error_stats.getMedian();

    clog::L(__func__, 1) << "Median error: " << calib.error_median << ", IQR: [" << error_stats.getQuantile(.25) << ", " << error_stats.getQuantile(.75) << "]";
    clog::L(__func__, 1) << "Parameters before: " << std::endl
                         << "Camera matrix: " << old_cam << std::endl
                         << "Distortion: " << old_dist << std::endl;
    clog::L(__func__, 1) << "Parameters after: " << std::endl
                         << "Camera matrix: " << calib.cameraMatrix << std::endl
                         << "Distortion: " << calib.distCoeffs << std::endl;
    clog::L(__func__, 1) << "Difference: old - new" << std::endl
                         << "Camera matrix: " << (old_cam - calib.cameraMatrix) << std::endl
                         << "Distortion: " << (old_dist - calib.distCoeffs) << std::endl;

    //CeresCalibKnownCorrections(outlier_threshold, calib);
    return 0;
}

template<int N>
double Calib::CeresCalibFlexibleTargetN(double const outlier_threshold) {
    // Build the problem.
    ceres::Problem problem;

    prepareCalibration();

    std::string const name = std::string("FlexibleN") + std::to_string(N);

    if (!hasCalibName(name)) {
        for (int d_N = 2; d_N < N; d_N += 2) {
            std::string prev = std::string("FlexibleN") + std::to_string(N - d_N);
            if (!hasCalibName(name) && hasCalibName(prev)) {
                calibrations[name] = calibrations[prev];
                if (calibrations[name].distN.size() != N+8) {
                    calibrations[name].distN.resize(N+8, 0.0);
                }
            }
        }
        if (!hasCalibName(name) && hasCalibName("Flexible")) {
            calibrations[name] = calibrations["Flexible"];
        }
        else if (!hasCalibName(name) && hasCalibName("Ceres")) {
            calibrations[name] = calibrations["Ceres"];
        }
        else if (!hasCalibName(name) && hasCalibName("OpenCV")) {
            calibrations[name] = calibrations["OpenCV"];
        }
    }

    CalibResult & calib = calibrations[name];
    calib.calib = this;
    calib.name = name;

    if (calib.distN.size() != N+8) {
        calib.distN = std::vector<double>(N+8, 0.0);
    }

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

        auto const & sub_data = data[imageFiles[ii]];
        size_t outlier_counter = 0;
        runningstats::QuantileStats<float> error_stats;
        for (size_t jj = 0; jj < imagePoints[ii].size(); ++jj) {
            cv::Scalar_<int> const c = this->ids[ii][jj];
            ids.insert(c);
            {
                FlexibleTargetProjectionFunctorN<N> loss (
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
                         calib.distN.data(),
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
                        FlexibleTargetProjectionFunctorN<N>,
                        2, // Number of residuals
                        1, // focal length x
                        1, // focal length y
                        1, // principal point x
                        1, // principal point y
                        3, // rotation vector for the target
                        3, // translation vector for the target
                        3, // correction vector for the 3d marker position
                        N+8 // distortion coefficients

                        >(new FlexibleTargetProjectionFunctorN<N> (
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
                                         calib.distN.data() // distortion coefficients
                                         );
            }
        }
        double const outlier_percentage = 100.0 * double(outlier_counter) / sub_data.size();
        outlier_percentages.push_unsafe(outlier_percentage);
        outlier_ranking.insert({outlier_percentage, imageFiles[ii] + " median: " + std::to_string(error_stats.getMedian())});
        calib.outlier_percentages[ii] = outlier_percentage;
    }
    clog::L(__func__ + std::to_string(N), 2) << "Outlier ranking:" << std::endl;
    for (auto const& it : outlier_ranking) {
        clog::L(__func__ + std::to_string(N), 2) << it.second << ": \t" << it.first << std::endl;
    }
    std::cout << "Ignored " << 100.0 * double(ignored_files_counter) / data.size() << "% of files" << std::endl;
    if (outlier_threshold > 0) {
        clog::L(__func__ + std::to_string(N), 2) << "Outlier percentage stats: " << outlier_percentages.print() << std::endl;
    }

    for (cv::Scalar_<int> const& it : ids) {
        ceres::CostFunction* cost_function =
                new ceres::AutoDiffCostFunction<LocalCorrectionsSum<3>, 3, 3>(
                    new LocalCorrectionsSum<3>(1));
        problem.AddResidualBlock(cost_function,
                                 nullptr, // Loss function (nullptr = L2)
                                 local_corrections[it].data() // correction
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

    clog::L(__func__ + std::to_string(N), 1) << summary.BriefReport() << "\n";
    clog::L(__func__ + std::to_string(N), 1) << summary.FullReport() << "\n";

    size_t counter = 0;

    for (auto const& it : local_corrections) {
        calib.objectPointCorrections[it.first] = vec2point3f(it.second);
    }

    for (size_t ii = 0; ii < local_rvecs.size(); ++ii) {
        calib.rvecs[ii] = vec2mat(local_rvecs[ii]);
        calib.tvecs[ii] = vec2mat(local_tvecs[ii]);
    }

    clog::L(__func__ + std::to_string(N), 1) << "Parameters before: " << std::endl
                                             << "Camera matrix: " << old_cam << std::endl
                                                ;//<< "Distortion: " << old_dist << std::endl;
    clog::L(__func__ + std::to_string(N), 1) << "Parameters after: " << std::endl
                                             << "Camera matrix: " << calib.cameraMatrix << std::endl
                                                ;//<< "Distortion: " << calib.distCoeffs << std::endl;
    clog::L(__func__ + std::to_string(N), 1) << "Difference: old - new" << std::endl
                                             << "Camera matrix: " << (old_cam - calib.cameraMatrix) << std::endl
                                                ;//<< "Distortion: " << (old_dist - calib.distCoeffs) << std::endl;

    //CeresCalibKnownCorrections(outlier_threshold, calib);
    return 0;
}

template double Calib::CeresCalibFlexibleTargetN< 2>(double const outlier_threshold);
template double Calib::CeresCalibFlexibleTargetN< 4>(double const outlier_threshold);
template double Calib::CeresCalibFlexibleTargetN< 6>(double const outlier_threshold);
template double Calib::CeresCalibFlexibleTargetN< 8>(double const outlier_threshold);
template double Calib::CeresCalibFlexibleTargetN<10>(double const outlier_threshold);
template double Calib::CeresCalibFlexibleTargetN<12>(double const outlier_threshold);
template double Calib::CeresCalibFlexibleTargetN<14>(double const outlier_threshold);
template double Calib::CeresCalibFlexibleTargetN<16>(double const outlier_threshold);

template<int N>
double Calib::CeresCalibFlexibleTargetOdd(double const outlier_threshold) {
    // Build the problem.
    ceres::Problem problem;

    prepareCalibration();

    std::string const name = std::string("FlexibleOdd") + std::to_string(N);

    if (!hasCalibName(name)) {
        for (int d_N = 2; d_N < N; d_N += 2) {
            std::string prev = std::string("FlexibleOdd") + std::to_string(N - d_N);
            if (!hasCalibName(name) && hasCalibName(prev)) {
                calibrations[name] = calibrations[prev];
                if (calibrations[name].distN.size() != N+8) {
                    calibrations[name].distN.resize(N+8, 0.0);
                }
            }
        }
        if (!hasCalibName(name) && hasCalibName("Flexible")) {
            calibrations[name] = calibrations["Flexible"];
        }
        else if (!hasCalibName(name) && hasCalibName("Ceres")) {
            calibrations[name] = calibrations["Ceres"];
        }
        else if (!hasCalibName(name) && hasCalibName("OpenCV")) {
            calibrations[name] = calibrations["OpenCV"];
        }
    }

    CalibResult & calib = calibrations[name];
    calib.calib = this;
    calib.name = name;

    if (calib.distN.size() != N+8) {
        calib.distN = std::vector<double>(N+8, 0.0);
    }

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

        auto const & sub_data = data[imageFiles[ii]];
        size_t outlier_counter = 0;
        runningstats::QuantileStats<float> error_stats;
        for (size_t jj = 0; jj < imagePoints[ii].size(); ++jj) {
            cv::Scalar_<int> const c = this->ids[ii][jj];
            ids.insert(c);
            {
                FlexibleTargetProjectionFunctorOdd<N> loss (
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
                         local_corrections[c].data(), // 3d point correction
                         &calib.x_factor,
                         calib.distN.data(),
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
                        FlexibleTargetProjectionFunctorOdd<N>,
                        2, // Number of residuals
                        1, // focal length x
                        1, // focal length y
                        1, // principal point x
                        1, // principal point y
                        3, // rotation vector for the target
                        3, // translation vector for the target
                        3, // correction vector for the 3d marker position
                        1,
                        N+8 // distortion coefficients

                        >(new FlexibleTargetProjectionFunctorOdd<N> (
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
                                         local_corrections[c].data(), // 3d point correction
                                         &calib.x_factor,
                                         calib.distN.data() // distortion coefficients
                                         );
            }
        }
        double const outlier_percentage = 100.0 * double(outlier_counter) / sub_data.size();
        outlier_percentages.push_unsafe(outlier_percentage);
        outlier_ranking.insert({outlier_percentage, imageFiles[ii] + " median: " + std::to_string(error_stats.getMedian())});
        calib.outlier_percentages[ii] = outlier_percentage;
    }
    if (outlier_threshold > 0) {
        clog::L(std::string(__func__) + std::to_string(N), 2) << "Outlier ranking:" << std::endl;
        for (auto const& it : outlier_ranking) {
            clog::L(std::string(__func__) + std::to_string(N), 2) << it.second << ": \t" << it.first << std::endl;
        }
        std::cout << "Ignored " << 100.0 * double(ignored_files_counter) / data.size() << "% of files" << std::endl;
        if (outlier_threshold > 0) {
            clog::L(std::string(__func__) + std::to_string(N), 2) << "Outlier percentage stats: " << outlier_percentages.print() << std::endl;
        }
    }

    for (cv::Scalar_<int> const& it : ids) {
        ceres::CostFunction* cost_function =
                new ceres::AutoDiffCostFunction<LocalCorrectionsSum<3>, 3, 3>(
                    new LocalCorrectionsSum<3>(1));
        problem.AddResidualBlock(cost_function,
                                 nullptr, // Loss function (nullptr = L2)
                                 local_corrections[it].data() // correction
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

    clog::L(std::string(__func__) + std::to_string(N), 1) << summary.BriefReport() << "\n";
    clog::L(std::string(__func__) + std::to_string(N), 1) << summary.FullReport() << "\n";

    for (auto const& it : local_corrections) {
        calib.objectPointCorrections[it.first] = vec2point3f(it.second);
    }

    for (size_t ii = 0; ii < local_rvecs.size(); ++ii) {
        calib.rvecs[ii] = vec2mat(local_rvecs[ii]);
        calib.tvecs[ii] = vec2mat(local_tvecs[ii]);
    }

    clog::L(std::string(__func__) + std::to_string(N), 1) << "Parameters before: " << std::endl
                                                          << "Camera matrix: " << old_cam << std::endl
                                                             ;//<< "Distortion: " << old_dist << std::endl;
    clog::L(std::string(__func__) + std::to_string(N), 1) << "Parameters after: " << std::endl
                                                          << "Camera matrix: " << calib.cameraMatrix << std::endl
                                                             ;//<< "Distortion: " << calib.distCoeffs << std::endl;
    clog::L(std::string(__func__) + std::to_string(N), 1) << "Difference: old - new" << std::endl
                                                          << "Camera matrix: " << (old_cam - calib.cameraMatrix) << std::endl
                                                             ;//<< "Distortion: " << (old_dist - calib.distCoeffs) << std::endl;
    clog::L(std::string(__func__) + std::to_string(N), 1) << "x_factor: " << calib.x_factor;
    //CeresCalibKnownCorrections(outlier_threshold, calib);
    return 0;
}

template double Calib::CeresCalibFlexibleTargetOdd< 2>(double const outlier_threshold);
template double Calib::CeresCalibFlexibleTargetOdd< 4>(double const outlier_threshold);
template double Calib::CeresCalibFlexibleTargetOdd< 6>(double const outlier_threshold);
template double Calib::CeresCalibFlexibleTargetOdd< 8>(double const outlier_threshold);
template double Calib::CeresCalibFlexibleTargetOdd<10>(double const outlier_threshold);
template double Calib::CeresCalibFlexibleTargetOdd<12>(double const outlier_threshold);
template double Calib::CeresCalibFlexibleTargetOdd<14>(double const outlier_threshold);
template double Calib::CeresCalibFlexibleTargetOdd<16>(double const outlier_threshold);



void Calib::minMaxPoint3f(cv::Point3f const& val, cv::Point3f& min, cv::Point3f& max) {
    min.x = std::min(val.x, min.x);
    min.y = std::min(val.y, min.y);
    min.z = std::min(val.z, min.z);

    max.x = std::max(val.x, max.x);
    max.y = std::max(val.y, max.y);
    max.z = std::max(val.z, max.z);
}

namespace {
void SetParameterBlockConstant(ceres::Problem& problem, std::vector<double*> const& blocks) {
    for (double * block : blocks) {
        problem.SetParameterBlockConstant(block);
    }
}
void SetParameterBlockVariable(ceres::Problem& problem, std::vector<double*> const& blocks) {
    for (double * block : blocks) {
        problem.SetParameterBlockVariable(block);
    }
}
}

double Calib::CeresCalibSemiFlexibleTarget(double const outlier_threshold) {
    // Build the problem.
    ceres::Problem problem;
    ceres::Problem dist_correction_problem;

    prepareCalibration();

    if (!hasCalibName("SemiFlexible")) {
        if (hasCalibName("Ceres")) {
            calibrations["SemiFlexible"] = calibrations["Ceres"];
        }
        else if (hasCalibName("OpenCV")) {
            calibrations["SemiFlexible"] = calibrations["OpenCV"];
        }
    }

    CalibResult & calib = calibrations["SemiFlexible"];
    calib.calib = this;
    calib.name = "SemiFlexible";

    std::vector<std::vector<double> > local_rvecs(data.size()), local_tvecs(data.size());

    std::vector<double> local_dist = mat2vec(calib.distCoeffs);
    if (local_dist.size() < 14) {
        local_dist.resize(14);
    }
    std::vector<double> local_inverseDistCoeffs = calib.inverseDistCoeffs;
    if (local_inverseDistCoeffs.size() < 14) {
        local_inverseDistCoeffs.resize(14);
    }

    cv::Mat_<double> old_cam = calib.cameraMatrix.clone();
    cv::Mat_<double> old_dist = calib.distCoeffs.clone();


    std::map<cv::Scalar_<int>, std::vector<double>, cmpScalar > local_corrections;

    cv::Point3f max_3d_pt{0,0,0};
    cv::Point3f min_3d_pt{std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),std::numeric_limits<float>::max()};

    for (const auto& it: data) {
        for (size_t ii = 0; ii < it.second.size(); ++ii) {
            cv::Scalar_<int> const c = getSimpleIdLayer(it.second.get(ii));
            local_corrections[c] = point2vec3f(calib.raw_objectPointCorrections[c]);
            minMaxPoint3f(getInitial3DCoord(it.second.get(ii)), min_3d_pt, max_3d_pt);
        }
    }
    cv::Point2f const target_center{(max_3d_pt.x + min_3d_pt.x)/2, (max_3d_pt.y + min_3d_pt.y)/2};

    std::set<cv::Scalar_<int>, cmpScalar> ids;
    std::vector<double*> traditional_blocks, object_correction_blocks;
    traditional_blocks.push_back(&calib.cameraMatrix(0,0));
    traditional_blocks.push_back(&calib.cameraMatrix(1,1));
    traditional_blocks.push_back(&calib.cameraMatrix(0,2));
    traditional_blocks.push_back(&calib.cameraMatrix(1,2));
    traditional_blocks.push_back(local_dist.data());
    if (calib.outlier_percentages.size() != data.size()) {
        calib.outlier_percentages = std::vector<double>(data.size(), 0.0);
    }
    size_t ignored_files_counter = 0;
    std::multimap<double, std::string> outlier_ranking;
    runningstats::RunningStats outlier_percentages;
    for (size_t ii = 0; ii < data.size(); ++ii) {
        local_rvecs[ii] = mat2vec(calib.rvecs[ii]);
        local_tvecs[ii] = mat2vec(calib.tvecs[ii]);
        traditional_blocks.push_back(local_rvecs[ii].data());
        traditional_blocks.push_back(local_tvecs[ii].data());
        bool ignore_current_file = false;
        if (calib.outlier_percentages[ii] > max_outlier_percentage) {
            ignore_current_file = true;
            ignored_files_counter++;
        }

        auto const & sub_data = data[imageFiles[ii]];
        size_t outlier_counter = 0;
        runningstats::QuantileStats<float> error_stats;
        for (size_t jj = 0; jj < imagePoints[ii].size(); ++jj) {
            cv::Scalar_<int> const c = this->ids[ii][jj];
            ids.insert(c);
            {
                SemiFlexibleTargetProjectionFunctor loss (
                            imagePoints[ii][jj],
                            objectPoints[ii][jj],
                            target_center
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
                         &calib.x_factor,
                         local_dist.data(),
                         local_inverseDistCoeffs.data(),
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
                        SemiFlexibleTargetProjectionFunctor,
                        2, // Number of residuals
                        1, // focal length x
                        1, // focal length y
                        1, // principal point x
                        1, // principal point y
                        3, // rotation vector for the target
                        3, // translation vector for the target
                        3, // correction vector for the 3d marker position
                        1, // x_factor
                        14, // distortion coefficients
                        14 // inverse target distortion correction

                        >(new SemiFlexibleTargetProjectionFunctor (
                              imagePoints[ii][jj],
                              objectPoints[ii][jj],
                              target_center
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
                                         &calib.x_factor,
                                         local_dist.data(), // distortion coefficients
                                         local_inverseDistCoeffs.data() // inverse target distortion correction
                                         );
            }
        }
        double const outlier_percentage = 100.0 * double(outlier_counter) / sub_data.size();
        outlier_percentages.push_unsafe(outlier_percentage);
        outlier_ranking.insert({outlier_percentage, imageFiles[ii] + " median: " + std::to_string(error_stats.getMedian())});
        calib.outlier_percentages[ii] = outlier_percentage;
    }
    if (outlier_threshold > 0) {
        clog::L(__func__, 2) << "Outlier ranking:" << std::endl;
        for (auto const& it : outlier_ranking) {
            clog::L(__func__, 2) << it.second << ": \t" << it.first << std::endl;
        }
        std::cout << "Ignored " << 100.0 * double(ignored_files_counter) / data.size() << "% of files" << std::endl;
        if (outlier_threshold > 0) {
            clog::L(__func__, 2) << "Outlier percentage stats: " << outlier_percentages.print() << std::endl;
        }
    }


    for (cv::Scalar_<int> const& it : ids) {
        {
            ceres::CostFunction* cost_function =
                    new ceres::AutoDiffCostFunction<LocalCorrectionsSum<3>, 3, 3>(
                        new LocalCorrectionsSum<3>(1));
            problem.AddResidualBlock(cost_function,
                                     nullptr, // Loss function (nullptr = L2)
                                     local_corrections[it].data() // correction
                                     );
        }
        {
            ceres::CostFunction* cost_function =
                    new ceres::AutoDiffCostFunction<DistortedTargetCorrectionFunctor, 2, 3, 14>(
                        new DistortedTargetCorrectionFunctor(getInitial3DCoord(it), target_center));
            problem.AddResidualBlock(cost_function,
                                     nullptr, // Loss function (nullptr = L2)
                                     local_corrections[it].data(), // correction
                                     local_inverseDistCoeffs.data()
                                     );
            object_correction_blocks.push_back(local_corrections[it].data());
        }
        {
            ceres::CostFunction* cost_function =
                    new ceres::AutoDiffCostFunction<DistortedTargetCorrectionFunctor, 2, 3, 14>(
                        new DistortedTargetCorrectionFunctor(getInitial3DCoord(it), target_center));
            dist_correction_problem.AddResidualBlock(cost_function,
                                                     nullptr, // Loss function (nullptr = L2)
                                                     local_corrections[it].data(), // correction
                                                     local_inverseDistCoeffs.data()
                                                     );
        }
    }

    // Run the solver!
    ceres::Solver::Options options;
    options.num_threads = int(threads);
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.max_num_iterations = max_iter;
    options.minimizer_progress_to_stdout = verbose;
    options.function_tolerance = 1e-18;
    options.gradient_tolerance = 1e-18;
    options.parameter_tolerance = 1e-18;

    {
        ceres::Solver::Summary summary;
        Solve(options, &problem, &summary);

        clog::L(__func__, 1) << summary.BriefReport() << "\n";
        clog::L(__func__, 1) << summary.FullReport() << "\n";
    }

    std::vector<cv::Vec2f> all_residuals;
    std::vector<Corner> all_features;
    size_t downscale_factor = 50;
    cv::Size downscaled_size(imageSize.width/downscale_factor+1, imageSize.height/downscale_factor+1);
    cv::Mat_<cv::Vec2f> residuals_image(downscaled_size, cv::Vec2f(0,0));
    cv::Mat_<int> residuals_count(downscaled_size, int(0));
    runningstats::QuantileStats<float> error_stats;
    for (size_t ii = 0; ii < data.size(); ++ii) {
        for (size_t jj = 0; jj < imagePoints[ii].size(); ++jj) {
            cv::Scalar_<int> const c = this->ids[ii][jj];
            {
                cv::Vec2d residuals(0,0);
                SemiFlexibleTargetProjectionFunctor functor(
                            imagePoints[ii][jj],
                            objectPoints[ii][jj],
                            target_center
                            );
                functor(&calib.cameraMatrix(0,0), // focal length x
                        &calib.cameraMatrix(1,1), // focal length y
                        &calib.cameraMatrix(0,2), // principal point x
                        &calib.cameraMatrix(1,2), // principal point y
                        local_rvecs[ii].data(), // rotation vector for the target
                        local_tvecs[ii].data(), // translation vector for the target
                        local_corrections[c].data(),
                        &calib.x_factor,
                        local_dist.data(), // distortion coefficients
                        local_inverseDistCoeffs.data(), // inverse target distortion correction
                        residuals.val
                        );
                error_stats.push_unsafe(cv::norm(residuals));
                cv::Point2i pt(std::round(imagePoints[ii][jj].x/downscale_factor), std::round(imagePoints[ii][jj].y/downscale_factor));
                residuals_image(pt) += residuals;
                residuals_count(pt)++;
                all_features.push_back(corners[ii][jj]);
                all_residuals.push_back(residuals);
            }
        }
    }
    saveReprojectionsJsons(cache_prefix + "-" + calib.name + "-reprojection-errors", all_features, all_residuals);
    for (int ii = 0; ii < residuals_image.rows; ++ii) {
        for (int jj = 0; jj < residuals_image.cols; ++jj) {
            if (residuals_count(ii, jj) > 0) {
                residuals_image(ii, jj) /= residuals_count(ii, jj);
            }
            else {
                residuals_image(ii, jj)[0] = residuals_image(ii, jj)[1] = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
    hdflow::gnuplotWithArrows("SemiFlexibleAfterCalib-residuals", residuals_image, -1, 100);
    clog::L(__func__, 2) << "Error stats: " << error_stats.print();
    clog::L(__func__, 2) << "Error median: " << error_stats.getMedian() << ", IQR: [" << error_stats.getQuantile(.25) << ", " << error_stats.getQuantile(.75) << "]";;

#if 0
    {
        ceres::Solver::Summary summary;
        SetParameterBlockConstant(dist_correction_problem, object_correction_blocks);
        Solve(options, &dist_correction_problem, &summary);

        clog::L(__func__, 1) << summary.BriefReport() << "\n";
        clog::L(__func__, 1) << summary.FullReport() << "\n";
    }
    {
        ceres::Solver::Summary summary;
        SetParameterBlockConstant(problem, object_correction_blocks);

        Solve(options, &problem, &summary);

        clog::L(__func__, 1) << summary.BriefReport() << "\n";
        clog::L(__func__, 1) << summary.FullReport() << "\n";
    }
#endif

    size_t counter = 0;
    for (auto & it : calib.distCoeffs) {
        it = local_dist[counter];
        counter++;
    }

    for (auto const& it : local_corrections) {
        cv::Vec3f src = vec2point3f(it.second);
        calib.raw_objectPointCorrections[it.first] = src;
        cv::Vec3f dst = vec2point3f(it.second);
        DistortedTargetCorrectionFunctor func(getInitial3DCoord(it.first), target_center);
        func(src.val, local_inverseDistCoeffs.data(), dst.val);
        calib.objectPointCorrections[it.first] = dst;
    }
    printObjectPointCorrectionsStats("SemiFlexibleRaw", calib.raw_objectPointCorrections);
    printObjectPointCorrectionsStats("SemiFlexibleCorrected", calib.objectPointCorrections);

    for (size_t ii = 0; ii < local_rvecs.size(); ++ii) {
        calib.rvecs[ii] = vec2mat(local_rvecs[ii]);
        calib.tvecs[ii] = vec2mat(local_tvecs[ii]);
    }

    clog::L(__func__, 1) << "x-factor: " << calib.x_factor;
    clog::L(__func__, 1) << "Parameters before: " << std::endl
                         << "Camera matrix: " << old_cam << std::endl
                         << "Distortion: " << old_dist << std::endl
                         << "Target undistortion: " << printVec(calib.inverseDistCoeffs);
    clog::L(__func__, 1) << "Parameters after: " << std::endl
                         << "Camera matrix: " << calib.cameraMatrix << std::endl
                         << "Distortion: " << calib.distCoeffs << std::endl
                         << "Target undistortion: " << printVec(local_inverseDistCoeffs);
    clog::L(__func__, 1) << "Difference: old - new" << std::endl
                         << "Camera matrix: " << (old_cam - calib.cameraMatrix) << std::endl
                         << "Distortion: " << (old_dist - calib.distCoeffs) << std::endl;
    calib.inverseDistCoeffs = local_inverseDistCoeffs;

    //CeresCalibKnownCorrections(outlier_threshold, calib);

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

/**
 * @brief startsWith checks if (str) starts with (term)
 * @param str
 * @param term
 * @return
 */
bool Calib::startsWith(std::string const& str, std::string const& term) {
    if (str.size() < term.size() || str.empty() || term.empty()) {
        return false;
    }
    return str.substr(0, term.size()) == term;
}

template<class F, class T>
/**
 * @brief project
 * @param name Name of the calibration result.
 * @param p 3D world point
 * @param result 2D projected point
 * @param focal focal lengths (f_x and f_y)
 * @param principal Principal point (orthogonal projection of the pinhole on the sensor)
 * @param R Rotation matrix (3x3)
 * @param t translation vector
 * @param dist distortion vector
 */
void CalibResult::projectByCalibName(
        F const p[3],
T result[2],
const T focal[2],
const T principal[2],
const T R[9],
const T t[3]) {
    for (std::string const prefix : {"FlexibleN", "SemiFlexibleN"}) {
        if (Calib::startsWith(name, prefix)) {
            int const N = std::stoi(name.substr(prefix.size()));
            assert(size_t(N+8) == distN.size());
            switch (N) {
            case  1: Calib::projectN< 1>(p, result, focal, principal, R, t, distN.data()); return;
            case  2: Calib::projectN< 2>(p, result, focal, principal, R, t, distN.data()); return;
            case  4: Calib::projectN< 4>(p, result, focal, principal, R, t, distN.data()); return;
            case  6: Calib::projectN< 6>(p, result, focal, principal, R, t, distN.data()); return;
            case  8: Calib::projectN< 8>(p, result, focal, principal, R, t, distN.data()); return;
            case 10: Calib::projectN<10>(p, result, focal, principal, R, t, distN.data()); return;
            case 12: Calib::projectN<12>(p, result, focal, principal, R, t, distN.data()); return;
            case 14: Calib::projectN<14>(p, result, focal, principal, R, t, distN.data()); return;
            case 16: Calib::projectN<16>(p, result, focal, principal, R, t, distN.data()); return;
            }
        }
    }
    for (std::string const prefix : {"FlexibleOdd", "SemiFlexibleOdd"}) {
        if (Calib::startsWith(name, prefix)) {
            int const N = std::stoi(name.substr(prefix.size()));
            assert(size_t(N+8) == distN.size());
            switch (N) {
            case  1: Calib::projectOdd< 1>(p, result, focal, principal, R, t, distN.data()); return;
            case  2: Calib::projectOdd< 2>(p, result, focal, principal, R, t, distN.data()); return;
            case  4: Calib::projectOdd< 4>(p, result, focal, principal, R, t, distN.data()); return;
            case  6: Calib::projectOdd< 6>(p, result, focal, principal, R, t, distN.data()); return;
            case  8: Calib::projectOdd< 8>(p, result, focal, principal, R, t, distN.data()); return;
            case 10: Calib::projectOdd<10>(p, result, focal, principal, R, t, distN.data()); return;
            case 12: Calib::projectOdd<12>(p, result, focal, principal, R, t, distN.data()); return;
            case 14: Calib::projectOdd<14>(p, result, focal, principal, R, t, distN.data()); return;
            case 16: Calib::projectOdd<16>(p, result, focal, principal, R, t, distN.data()); return;
            }
        }
    }
    for (std::string const prefix : {"Spline-"}) {
        if (Calib::startsWith(name, prefix)) {
            int const N = std::stoi(name.substr(prefix.size()));
            if (size_t((N+3)*(N+3)) != spline_x.size()) {
                throw std::runtime_error(
                            std::string("Size of spline doesn't match: N=")
                            + std::to_string(N)
                            + ", expected: " + std::to_string((N+3)*(N+3))
                            + ", got instead: " + std::to_string(spline_x.size()));
            }
            assert(size_t((N+3)*(N+3)) == spline_x.size());
            assert(size_t((N+3)*(N+3)) == spline_y.size());
            switch (N) {
            case  3: Calib::projectSpline< 3, 3>(p, result, focal, principal, R, t, spline_x.data(), spline_y.data(), calib->getImageSize()); return;
            case  5: Calib::projectSpline< 5, 3>(p, result, focal, principal, R, t, spline_x.data(), spline_y.data(), calib->getImageSize()); return;
            case  7: Calib::projectSpline< 7, 3>(p, result, focal, principal, R, t, spline_x.data(), spline_y.data(), calib->getImageSize()); return;
            case  9: Calib::projectSpline< 9, 3>(p, result, focal, principal, R, t, spline_x.data(), spline_y.data(), calib->getImageSize()); return;
            case 11: Calib::projectSpline<11, 3>(p, result, focal, principal, R, t, spline_x.data(), spline_y.data(), calib->getImageSize()); return;
            case 13: Calib::projectSpline<13, 3>(p, result, focal, principal, R, t, spline_x.data(), spline_y.data(), calib->getImageSize()); return;
            case 15: Calib::projectSpline<15, 3>(p, result, focal, principal, R, t, spline_x.data(), spline_y.data(), calib->getImageSize()); return;
            case 17: Calib::projectSpline<17, 3>(p, result, focal, principal, R, t, spline_x.data(), spline_y.data(), calib->getImageSize()); return;
            case 19: Calib::projectSpline<19, 3>(p, result, focal, principal, R, t, spline_x.data(), spline_y.data(), calib->getImageSize()); return;
            case 21: Calib::projectSpline<21, 3>(p, result, focal, principal, R, t, spline_x.data(), spline_y.data(), calib->getImageSize()); return;
            }
        }
    }
    std::vector<double> dist = getDistCoeffsVector();
    Calib::project(p, result, focal, principal, R, t, dist.data());
}

void CalibResult::getReprojections(
        const size_t ii,
        std::vector<Corner> &markers,
        std::vector<Point2d> &reprojections) {
    if (nullptr == calib) {
        throw std::runtime_error("Calib " + name + " doesn't have a pointer to the corresponding calibration data object (of class Calib)");
    }

    calib->prepareCalibrationByName(name);
    assert(ii < calib->data.size());

    std::vector<cv::Point2f> imgPoints = calib->getImagePoints()[ii];
    std::vector<cv::Point3f> objPoints = calib->getObjectPoints()[ii];
    assert(imgPoints.size() == objPoints.size());
    std::string const& filename = imageFiles[ii];
    CornerStore const& store = calib->data[filename];
    cv::Mat const& rvec = rvecs[ii];
    cv::Mat const& tvec = tvecs[ii];
    assert(rvecs.size() == tvecs.size());

    markers.clear();
    markers.reserve(imgPoints.size());

    reprojections.clear();
    reprojections.reserve(imgPoints.size());

    double p[3], r[3], t[3], R[9];

    double result[2];

    cv::Point3d r_point(rvec);
    cv::Point3d t_point(tvec);

    vec2arr(r, r_point);
    vec2arr(t, t_point);

    Calib::rot_vec2mat(r, R);

    double focal[2] = {cameraMatrix(0,0), cameraMatrix(1,1)};
    double principal[2] = {cameraMatrix(0,2), cameraMatrix(1,2)};

    std::vector<std::vector<double> > data;
    for (size_t jj = 0; jj < imgPoints.size(); ++jj) {
        Corner const& current_corner = store.get(jj);
        cv::Scalar_<int> simple_id = Calib::getSimpleIdLayer(current_corner);
        cv::Point3f correction(0,0,0);
        auto const it = objectPointCorrections.find(simple_id);
        if (objectPointCorrections.end() != it) {
            correction = it->second;
        }
        objPoints[ii].x *= (1.0 + x_factor/100);
        cv::Point3f current_objPoint = objPoints[jj] + correction;

        vec2arr(p, current_objPoint);
        projectByCalibName(p, result, focal, principal, R, t);
        cv::Point2d res(result[0], result[1]);
        markers.push_back(current_corner);
        reprojections.push_back(res);
    }
}

void CalibResult::getAllReprojections(std::vector<Corner> &markers,
                                      std::vector<Point2d> &reprojections) {
    assert(markers.size() == reprojections.size());
    for (size_t ii = 0; ii < calib->data.size(); ++ii) {
        std::vector<Corner> local_markers;
        std::vector<Point2d> local_reprojections;
        getReprojections(ii, local_markers, local_reprojections);
        assert(local_markers.size() == local_reprojections.size());
        markers.insert(markers.end(), local_markers.begin(), local_markers.end());
        reprojections.insert(reprojections.end(), local_reprojections.begin(), local_reprojections.end());
        assert(markers.size() == reprojections.size());
    }

    assert(markers.size() == reprojections.size());
    errors.clear();
    for (size_t ii = 0; ii < markers.size(); ++ii) {
        double dist = cv::norm(cv::Point2d(markers[ii].p) - reprojections[ii]);
        errors.push_unsafe(dist);
    }
    error_percentiles.clear();
    for (double t = 0; t <= 1.000001; t += 0.01) {
        error_percentiles.push_back(errors.getQuantile(t));
    }
    error_median = errors.getMedian();
}

double CalibResult::getErrorMedian() {
    if (error_median <= 0) {
        std::vector<cv::Point2d> reprojections;
        std::vector<Corner> markers;
        getAllReprojections(markers, reprojections);
    }
    assert(error_median >= 0);
    return error_median;
}

std::vector<double> CalibResult::getDistCoeffsVector() {
    std::vector<double> result(std::max(14, distCoeffs.rows * distCoeffs.cols), 0.0);
    for (int ii = 0; ii < distCoeffs.rows * distCoeffs.cols; ++ii) {
        result[size_t(ii)] = distCoeffs(ii);
    }
    return result;
}

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

template<int N, class F, class T>
void Calib::projectN(
        F const p[3],
T result[2],
const T focal[2],
const T principal[2],
const T R[9],
const T t[3],
const T dist[N+8]
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
    T const& p1 = dist[N];
    T const& p2 = dist[N+1];
    T const& s1 = dist[N+2];
    T const& s2 = dist[N+3];
    T const& s3 = dist[N+4];
    T const& s4 = dist[N+5];
    T const& tau_x = dist[N+6];
    T const& tau_y = dist[N+7];

    T const r2 = x*x + y*y;
    T const r = ceres::sqrt(r2);
    T const r4 = r2*r2;

    T top(1);
    T bottom(1);
    T current_r = r;
    for (size_t ii = 0; ii < N; ii += 2) {
        top    += dist[ii+0] * current_r;
        bottom += dist[ii+1] * current_r;
        current_r *= r;
    }
    T const radial = top/bottom;

    T x2 = x*radial
            + T(2)*x*y*p1 + p2*(r2 + T(2)*x*x) + s1*r2 + s2*r4;

    T y2 = y*radial
            + T(2)*x*y*p2 + p1*(r2 + T(2)*y*y) + s3*r2 + s4*r4;

    applySensorTilt(x2, y2, tau_x, tau_y);

    x = x2 * focal[0] + principal[0];
    y = y2 * focal[1] + principal[1];
}

template<int N, class F, class T>
void Calib::projectOdd(
        F const p[3],
T result[2],
const T focal[2],
const T principal[2],
const T R[9],
const T t[3],
const T dist[N+8]
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
    T const& p1 = dist[N];
    T const& p2 = dist[N+1];
    T const& s1 = dist[N+2];
    T const& s2 = dist[N+3];
    T const& s3 = dist[N+4];
    T const& s4 = dist[N+5];
    T const& tau_x = dist[N+6];
    T const& tau_y = dist[N+7];

    T const r2 = x*x + y*y;
    T const r4 = r2*r2;

    T top(1);
    T bottom(1);
    T current_r = r2;
    for (size_t ii = 0; ii < N; ii += 2) {
        top    += dist[ii+0] * current_r;
        bottom += dist[ii+1] * current_r;
        current_r *= r2;
    }
    T const radial = top/bottom;

    T x2 = x*radial
            + T(2)*x*y*p1 + p2*(r2 + T(2)*x*x) + s1*r2 + s2*r4;

    T y2 = y*radial
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
{}

template<int N>
FlexibleTargetProjectionFunctorN<N>::FlexibleTargetProjectionFunctorN(
        const Point2f &_marker,
        const Point3f &_point): marker(_marker), point(_point)
{}

template<int N>
FlexibleTargetProjectionFunctorOdd<N>::FlexibleTargetProjectionFunctorOdd(
        const Point2f &_marker,
        const Point3f &_point): marker(_marker), point(_point)
{}

template<class T>
bool FlexibleTargetProjectionFunctor::operator()(const T * const f_x,
                                                 const T * const f_y,
                                                 const T * const c_x,
                                                 const T * const c_y,
                                                 const T * const rvec,
                                                 const T * const tvec,
                                                 const T * const correction,
                                                 const T * const x_factor,
                                                 const T * const dist,
                                                 T *residuals) const
{
    T rot_mat[9];
    Calib::rot_vec2mat(rvec, rot_mat);
    T const f[2] = {f_x[0], f_y[0]};
    T const c[2] = {c_x[0], c_y[0]};
    T result[2] = {T(0), T(0)};
    T const src[3] = {T(point.x)*(T(1) + x_factor[0]/100.0) + correction[0],
                      T(point.y) + correction[1],
                      T(point.z) + correction[2]};
    Calib::project(src, result, f, c, rot_mat, tvec, dist);
    residuals[0] = T(weight) * (result[0] - T(marker.x));
    residuals[1] = T(weight) * (result[1] - T(marker.y));
    return true;
}

template<int N>
template<class T>
bool FlexibleTargetProjectionFunctorN<N>::operator()(
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
    Calib::projectN<N>(src, result, f, c, rot_mat, tvec, dist);
    residuals[0] = T(weight) * (result[0] - T(marker.x));
    residuals[1] = T(weight) * (result[1] - T(marker.y));
    return true;
}

template<int N>
template<class T>
bool FlexibleTargetProjectionFunctorOdd<N>::operator()(const T * const f_x,
                                                       const T * const f_y,
                                                       const T * const c_x,
                                                       const T * const c_y,
                                                       const T * const rvec,
                                                       const T * const tvec,
                                                       const T * const correction,
                                                       const T * const x_factor,
                                                       const T * const dist,
                                                       T *residuals) const
{
    T rot_mat[9];
    Calib::rot_vec2mat(rvec, rot_mat);
    T const f[2] = {f_x[0], f_y[0]};
    T const c[2] = {c_x[0], c_y[0]};
    T result[2] = {T(0), T(0)};
    T const src[3] = {T(point.x)*(T(1) + x_factor[0]/100.0) + correction[0],
                      T(point.y) + correction[1],
                      T(point.z) + correction[2]};
    Calib::projectOdd<N>(src, result, f, c, rot_mat, tvec, dist);
    residuals[0] = T(weight) * (result[0] - T(marker.x));
    residuals[1] = T(weight) * (result[1] - T(marker.y));
    return true;
}

SingleProjectionFunctor::SingleProjectionFunctor(
        const Point2f &_marker,
        const Point3f &_point): marker(_marker), point(_point)
{}

template<class T>
bool SingleProjectionFunctor::operator()(
        const T * const f_x,
        const T * const f_y,
        const T * const c_x,
        const T * const c_y,
        const T * const rvec,
        const T * const tvec,
        const T * const dist,
        T *residuals) const
{
    T rot_mat[9];
    Calib::rot_vec2mat(rvec, rot_mat);
    T const f[2] = {f_x[0], f_y[0]};
    T const c[2] = {c_x[0], c_y[0]};
    T result[2] = {T(0), T(0)};
    T const src[3] = {T(point.x),
                      T(point.y),
                      T(point.z)};
    Calib::project(src, result, f, c, rot_mat, tvec, dist);
    residuals[0] = T(weight) * (result[0] - T(marker.x));
    residuals[1] = T(weight) * (result[1] - T(marker.y));
    return true;
}


SemiFlexibleTargetProjectionFunctor::SemiFlexibleTargetProjectionFunctor(const Point2f &_marker,
                                                                         const Point3f &_point, const Point2f &_center): marker(_marker), point(_point), center(_center)
{}

template<class T, class U>
void SemiFlexibleTargetProjectionFunctor::applyInverseDist(T const src[2],
T dst[2],
T const center[2],
const U dist[]
) {

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

    T const x = src[0] - center[0];
    T const y = src[1] - center[1];

    T const r2 = x*x + y*y;
    T const r4 = r2*r2;
    T const r6 = r4*r2;

    T & x2 = dst[0];
    x2 = x*(T(1) + k1*r2 + k2*r4 + k3*r6)/(T(1) + k4*r2 + k5*r4 + k6*r6)
            + T(2)*x*y*p1 + p2*(r2 + T(2)*x*x) + s1*r2 + s2*r4;

    T & y2 = dst[1];
    y2 = y*(T(1) + k1*r2 + k2*r4 + k3*r6)/(T(1) + k4*r2 + k5*r4 + k6*r6)
            + T(2)*x*y*p2 + p1*(r2 + T(2)*y*y) + s3*r2 + s4*r4;

    applySensorTilt(x2, y2, tau_x, tau_y);

    x2 += center[0];
    y2 += center[1];

}

template<class T>
bool SemiFlexibleTargetProjectionFunctor::operator()(
        const T * const f_x,
        const T * const f_y,
        const T * const c_x,
        const T * const c_y,
        const T * const rvec,
        const T * const tvec,
        const T * const correction,
        const T * const x_factor,
        const T * const dist,
        const T * const inverse_dist,
        T *residuals) const
{
    T rot_mat[9];
    Calib::rot_vec2mat(rvec, rot_mat);
    T const f[2] = {f_x[0], f_y[0]};
    T const c[2] = {c_x[0], c_y[0]};
    T result[2] = {T(0), T(0)};
    T const src[3] = {T(point.x)*(T(1) + x_factor[0]/100.0) + correction[0],
                      T(point.y) + correction[1],
                      T(point.z) + correction[2]};
    T dst[3] = {T(point.x) + correction[0],
                T(point.y) + correction[1],
                T(point.z) + correction[2]};
    T _center[2] = {T(center.x), T(center.y)};
    applyInverseDist(src, dst, _center, inverse_dist);
    Calib::project(src, result, f, c, rot_mat, tvec, dist);
    residuals[0] = T(weight) * (result[0] - T(marker.x));
    residuals[1] = T(weight) * (result[1] - T(marker.y));
    return true;
}

DistortedTargetCorrectionFunctor::DistortedTargetCorrectionFunctor(const Point3f &_src, const Point2f &_center)
    : src(_src), center(_center) {}

template<class T, class U>
bool DistortedTargetCorrectionFunctor::operator()(const T * const correction,
                                                  const U * const inverse_dist,
                                                  T *residuals) const {
    T const _src[2] = {T(src.x) + correction[0], T(src.y) + correction[1]};
    T corrected[2] = {T(0), T(0)};
    T const _center[2] = {T(this->center.x), T(this->center.y)};
    SemiFlexibleTargetProjectionFunctor::applyInverseDist(_src, corrected, _center, inverse_dist);
    residuals[0] = corrected[0] - T(src.x);
    residuals[1] = corrected[1] - T(src.y);
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
