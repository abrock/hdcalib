#include "hdcalib.h"

#include <ceres/ceres.h>
#include <opencv2/optflow.hpp>
#include <opencv2/video/tracking.hpp>
#include <runningstats/runningstats.h>
#include <catlogger/catlogger.h>
#include "gnuplot-iostream.h"

#undef NDEBUG
#include <assert.h>   // reinclude the header to update the definition of assert()

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

struct CornerIdSort
{
    inline bool operator() (const hdmarker::Corner& a, const hdmarker::Corner& b)
    {
        if (a.page < b.page) return true;
        if (a.page > b.page) return false;

        if (a.id.x < b.id.x) return true;
        if (a.id.x > b.id.x) return false;

        if (a.id.y < b.id.y) return true;
        if (a.id.y > b.id.y) return false;
        return false;
    }
};

bool Calib::removeOutliers(std::string const& calibName, const double threshold) {
    prepareCalibration();
    std::vector<hdmarker::Corner> outliers;
    for (size_t ii = 0; ii < data.size(); ++ii) {
        findOutliers(calibName,
                     threshold,
                     ii,
                     outliers);
    }
    if (outliers.empty()) {
        return false;
    }
    CornerStore subtrahend(outliers);
    //clog::L(__func__, 2) << "Outlier percentage by image:" << std::endl;
    runningstats::RunningStats percent_stats;
    for (auto& it : data) {
        size_t const before = it.second.size();
        it.second.difference(subtrahend);
        size_t const after = it.second.size();
        double const percent = (double(before - after)/before)*100.0;
        percent_stats.push(percent);
        //clog::L(__func__, 2) << it.first << ": removed " << (before-after) << " out of " << before << " corners (" << percent << "%)" << std::endl;
    }
    clog::L(__func__, 1) << "Removal percentage stats: " << percent_stats.print() << std::endl;
    invalidateCache();
    return true;
}

bool Calib::removeAllOutliers(const double threshold) {
    bool result = false;
    for (auto const& it : calibrations) {
        result = removeOutliers(it.first, threshold) | result;
    }
    return result;
}

void Calib::getReprojections(
        CalibResult & calib,
        const size_t ii,
        std::vector<Point2d> &markers,
        std::vector<Point2d> &reprojections) {

    auto const& imgPoints = imagePoints[ii];
    auto const& objPoints = objectPoints[ii];
    std::string const& filename = imageFiles[ii];
    CornerStore const& store = data[filename];
    cv::Mat const& rvec = calib.rvecs[ii];
    cv::Mat const& tvec = calib.tvecs[ii];

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

    rot_vec2mat(r, R);

    double dist[14];
    for (int jj = 0; jj < 14; ++jj) {
        dist[size_t(jj)] = calib.distCoeffs.cols > jj ? calib.distCoeffs(jj) : 0;
    }

    double focal[2] = {calib.cameraMatrix(0,0), calib.cameraMatrix(1,1)};
    double principal[2] = {calib.cameraMatrix(0,2), calib.cameraMatrix(1,2)};

    std::vector<std::vector<double> > data;
    for (size_t jj = 0; jj < imgPoints.size(); ++jj) {
        Corner const& current_corner = store.get(jj);
        cv::Point3i simple_id = getSimpleId(current_corner);
        cv::Point3f correction;
        auto const it = calib.objectPointCorrections.find(simple_id);
        if (calib.objectPointCorrections.end() != it) {
            correction = it->second;
        }
        cv::Point3f current_objPoint = objPoints[jj] + correction;

        vec2arr(p, current_objPoint);
        cv::Point2d marker_pos(imgPoints[jj]);
        project(p, result, focal, principal, R, t, dist);
        cv::Point2d res(result[0], result[1]);
        markers.push_back(marker_pos);
        reprojections.push_back(res);
    }
}

char Calib::color(const int ii, const int jj) {
    if (0 == (ii % 2)) { // First row
        if (0 == (jj % 2)) { // First column (top left pixel)
            return 'R';
        }
        return 'G'; // Top right pixel
    }
    if (0 == (jj % 2)) { // Second row, first column (bottom left pixel)
        return 'G';
    }
    return 'B'; // Second row, second pixel (bottom right pixel)
}

Calib::Calib() {
    clog::L(__func__, 2) << "Number of concurrent threads: " << threads << std::endl;
}

void Calib::save(const string &filename) {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    fs << "calibration" << *this;
    fs.release();}

std::vector<string> Calib::getImageNames() const {
    std::vector<std::string> result;
    for (const auto& it : data) {
        result.push_back(it.first);
    }
    return result;
}

void Calib::purgeSubmarkers() {
    CornerStore union_store;
    for (auto& it : data) {
        CornerStore & store = it.second;
        store.replaceCorners(store.getMainMarkers(cornerIdFactor));
        union_store.addConditional(store.getCorners());
    }
    for (std::pair<const std::string, CalibResult> & c : calibrations) {
        c.second.keepMarkers(union_store);
    }
}

Point3f Calib::getTransformedPoint(CalibResult &res, const string &filename, const Point3f &pt) {
    cv::Vec3d result = get3DPointWithoutCorrection(pt, res.getRVec(filename), res.getTVec(filename));
    return cv::Point3f(result[0], result[1], result[2]);
}

void Calib::checkSamePosition(const std::vector<string> &suffixes, const string calibration_type) {
    CalibResult & calib = getCalib(calibration_type);

    prepareCalibration();

    std::map<std::string, std::vector<std::string> > comparisons = matchSuffixes(imageFiles, suffixes);

    std::multimap<double, std::string> errors;

    for (std::pair<const std::string, std::vector<std::string> > const& it : comparisons) {
        if (it.second.size() < 2) {
            continue;
        }
        std::string const& prefix = it.first;
        std::vector<std::string> const& files = it.second;

        std::vector<std::vector<Vec3d> > pts = getCommon3DPoints(calib, files);
        assert(pts.size() == files.size());
        runningstats::QuantileStats<float> diff_stats;
        for (size_t ref_index = 0; ref_index < pts.size(); ++ref_index) {
            std::vector<cv::Vec3d> const& ref = pts[ref_index];
            for (size_t comp_index = ref_index+1; comp_index < pts.size(); comp_index++) {
                std::vector<cv::Vec3d> const& comp = pts[comp_index];
                assert(ref.size() == comp.size());
                for (size_t ii = 0; ii < ref.size() && ii < comp.size(); ++ii) {
                    cv::Vec3d residual = ref[ii] - comp[ii];
                    double const diff = std::sqrt(residual.dot(residual));
                    diff_stats.push_unsafe(diff);
                }
            }
        }
        double const median = diff_stats.getMedian();
        errors.insert({median, prefix});
    }
    clog::L(__func__, 2) << "Error stats:" << std::endl;
    for (std::pair<const double, std::string> const& it : errors) {
        clog::L(__func__, 2) << it.first << "\t" << it.second << std::endl;
    }
}

void Calib::checkSamePosition2D(const std::vector<string> &suffixes) {
    std::map<std::string, std::vector<std::string> > comparisons = matchSuffixes(imageFiles, suffixes);

    std::multimap<double, std::string> errors;
    for (std::pair<const std::string, std::vector<std::string> > const& it : comparisons) {
        if (it.second.size() < 2) {
            continue;
        }
        std::string const& prefix = it.first;
        std::vector<std::string> const& files = it.second;
        runningstats::QuantileStats<float> diff_stats;
        for (size_t ref_index = 0; ref_index < files.size(); ++ref_index) {
            CornerStore & ref = data[files[ref_index]];
            for (size_t comp_index = ref_index+1; comp_index < files.size(); comp_index++) {
                CornerStore & comp = data[files[comp_index]];
                for (size_t ii = 0; ii < ref.size(); ++ii) {
                    hdmarker::Corner const ref_p = ref.get(ii);
                    std::vector<hdmarker::Corner> comp_p = comp.findByID(ref_p);
                    if (comp_p.empty()) {
                        continue;
                    }
                    cv::Point2f residual = ref_p.p - comp_p[0].p;
                    double const dist = std::sqrt(residual.dot(residual));
                    diff_stats.push_unsafe(dist);
                }
            }
        }
        double const median = diff_stats.getMedian();
        errors.insert({median, prefix});
    }
    clog::L(__func__, 2) << "Error stats:" << std::endl;
    for (std::pair<const double, std::string> const& it : errors) {
        clog::L(__func__, 2) << it.first << "\t" << it.second << std::endl;
    }
}

std::vector<std::vector<Vec3d> > Calib::getCommon3DPoints(CalibResult &calib, const std::vector<string> &files) {
    std::vector<std::vector<cv::Vec3d> > result(files.size());
    if (files.size() < 2) {
        return result;
    }
    CornerStore intersection = data[files[0]];
    std::vector<CornerStore> stores;
    for (size_t ii = 0; ii < files.size(); ++ii) {
        stores.push_back(data[files[ii]]);
        intersection.intersect(data[files[ii]]);
    }
    for (size_t ii = 0; ii < stores.size(); ++ii) {
        CornerStore & store = stores[ii];
        store.intersect(intersection);
        assert(store.size() == intersection.size());
        size_t const file_index = getId(files[ii]);
        for (size_t jj = 0; jj < store.size(); ++jj) {
            result[ii].push_back(get3DPoint(calib, store.get(jj), calib.rvecs[file_index], calib.tvecs[file_index]));
        }
    }
    return result;
}

std::map<string, std::vector<string> > Calib::matchSuffixes(const std::vector<string> &images, const std::vector<string> &suffixes) {
    std::map<std::string, std::vector<std::string> > result;
    for (std::string const & image : images) {
        for (std::string const& suff : suffixes) {
            if (suff.size() > image.size()) {
                continue;
            }
            if (suff == image.substr(image.size() - suff.size())) {
                std::string const prefix = image.substr(0, image.size() - suff.size());
                result[prefix].push_back(image);
            }
        }
    }
    return result;
}

void Calib::setMaxOutlierPercentage(const double new_val) {
    max_outlier_percentage = new_val;
}

void Calib::deleteCalib(const string name) {
    auto it = calibrations.find(name);
    if (it != calibrations.end()) {
        calibrations.erase(it);
    }
}

void Calib::setCauchyParam(const double new_val) {
    cauchy_param = new_val;
}

void Calib::plotResidualsIntoImages(const string calib_name) {
    CalibResult & calib = getCalib(calib_name);

    std::cout << "Plot residuals into images" << std::endl;
    std::cout << std::string(imageFiles.size(), '-') << std::endl;
#pragma omp parallel for schedule(dynamic)
    for (size_t ii = 0; ii < imageFiles.size(); ++ii) {
        std::vector<cv::Point2d> markers, reprojections;
        getReprojections(calib, ii, markers, reprojections);
        cv::Mat img = readImage(imageFiles[ii], demosaic, libraw, useOnlyGreen);
        if (1 == img.channels()) {
            cv::Mat _img[3] = {img.clone(), img.clone(), img.clone()};
            cv::merge(_img, 3, img);
        }
        for (size_t jj = 0; jj < markers.size(); ++jj) {
            cv::Point2d const & marker = markers[jj];
            cv::Point2d const & repr = reprojections[jj];
            cv::Point2d const residual = marker - repr;
            double const err_sq = residual.dot(residual);
            cv::circle(img, marker, 2, cv::Scalar(0,0,255), cv::FILLED);
            cv::Scalar line_color = err_sq > outlier_threshold * outlier_threshold ? cv::Scalar(0,255,255) : cv::Scalar(0,255,0);
            cv::Scalar target_color = err_sq > outlier_threshold * outlier_threshold ? cv::Scalar(0,255,255) : cv::Scalar(0,255,0);
            cv::circle(img, repr, 2, target_color, cv::FILLED);
            cv::line(img, marker, repr, line_color, 3, cv::LINE_AA);
        }
        cv::imwrite(imageFiles[ii] + "-repr.png", img);
        std::cout << '.' << std::flush;
    }
    std::cout << std::endl;
}

void Calib::setOutlierThreshold(const double new_val) {
    outlier_threshold = new_val;
}

void Calib::purgeUnlikelyByDetectedRectangles() {
    cv::Rect_<int> const limits = getIdRectangleUnion();
    clog::L(__func__, 2) << "Purging unlikely markers by detected rectangles:" << std::endl;
    std::cout << std::string(data.size(), '-') << std::endl;
    for (auto & it : data) {
        CornerStore & store = it.second;
        store.purgeOutOfBounds(limits.x, limits.y, limits.x + limits.width, limits.y + limits.height);
        std::cout << '.' << std::flush;
    }
    std::cout << std::endl;
}

Rect_<int> Calib::getIdRectangleUnion() const {
    int min_x = std::numeric_limits<int>::max();
    int min_y = std::numeric_limits<int>::max();
    int max_x = 0;
    int max_y = 0;
    int const _cornerIdFactor = computeCornerIdFactor(recursionDepth);
    for (auto const& it : data) {
        CornerStore const & store = it.second;
        std::vector<hdmarker::Corner> const main_markers = store.getMainMarkers(_cornerIdFactor);
        for (hdmarker::Corner const& c : main_markers) {
            min_x = std::min(min_x, c.id.x);
            min_y = std::min(min_y, c.id.y);
            max_x = std::max(max_x, c.id.x);
            max_y = std::max(max_y, c.id.y);
        }
    }
    clog::L(__func__, 2) << "Rect limit: (" << min_x << ", " << min_y << ") / (" << max_x << ", " << max_y << ")" << std::endl;
    return cv::Rect_<int>(min_x, min_y, max_x - min_x, max_y - min_y);
}

double Calib::distance(const Corner &a, const Corner &b) {
    cv::Point2f res = a.p - b.p;
    return std::sqrt(res.dot(res));
}

void Calib::plotPoly(cv::Mat & img, std::vector<cv::Point> const& poly, cv::Scalar const& color, int const line) {
    int num_points = poly.size();
    cv::Point const * points = poly.data();
    cv::fillPoly(img, &points, &num_points, 1, color, line);
}

cv::Mat_<uint8_t> Calib::getMainMarkersArea(const std::vector<Corner> &submarkers, cv::Scalar const color, int const line) {
    cv::Mat_<uint8_t> result(imageSize, uint8_t(0));

    CornerStore store(submarkers);
    int const factor = getCornerIdFactor();

    std::vector<std::vector<hdmarker::Corner> > const squares = store.getSquares(factor);

    for (auto const& c : squares) {
        std::vector<cv::Point> poly;
        for (auto const& marker: c) {
            poly.push_back(marker.p);
        }
        plotPoly(result, poly, color, line);
    }
    /*
    for (auto const& c : squares) {
        std::vector<cv::Point> poly;
        if (c[0].id.y == factor) {
            poly.push_back(c[0].p);
            poly.push_back(2*c[0].p -c[1].p);
            poly.push_back(2*c[3].p -c[2].p);
            poly.push_back(c[3].p);
            plotPoly(result, poly, cv::Scalar(127), line);
        }
    }
    for (auto const& c : squares) {
        std::vector<cv::Point> poly;
        if (c[0].id.x == factor) {
            poly.push_back(c[0].p);
            poly.push_back(2*c[0].p -c[3].p);
            poly.push_back(2*c[1].p -c[2].p);
            poly.push_back(c[1].p);
            plotPoly(result, poly, cv::Scalar(127), line);
        }
    }
    */
    return result;
}

void Calib::exportPointClouds(const string &calib_name, double const outlier_threshold) {
    if (!hasCalibName(calib_name)) {
        throw std::runtime_error(std::string("Calib name ") + calib_name + " not available in exportPointClouds.");
    }
    prepareCalibration();
    CalibResult & calib = calibrations[calib_name];

    std::vector<cv::Point2d> markers, reprojections;
    for (size_t ii = 0; ii < imageFiles.size(); ++ii) {
        std::string const& filename = imageFiles[ii];
        std::ofstream out(filename + "-" + calib_name + "-target-points.txt", std::ofstream::out);
        CornerStore const& store = data[filename];
        getReprojections(calib, ii, markers, reprojections);
        if (markers.size() != reprojections.size() || markers.size() != store.size()) {
            throw std::runtime_error("Vector sizes don't match in exportPointClouds");
        }
        for (size_t jj = 0; jj < store.size(); ++jj) {
            if (outlier_threshold <= 0 || distance(markers[jj], reprojections[jj]) < outlier_threshold) {
                cv::Vec3d p = get3DPoint(calib, store.get(jj), calib.rvecs[ii], calib.tvecs[ii]);
                out << p[0] << "\t" << p[1] << "\t" << p[2] << std::endl;
            }
        }
    }
}

Mat Calib::calculateUndistortRectifyMap(CalibResult & calib) {
    clog::L(__func__, 2) << "initial cameraMatrix: " << calib.cameraMatrix << std::endl;

    cv::Mat_<double> newCameraMatrix = calib.cameraMatrix.clone();
    double const focal = std::sqrt(newCameraMatrix(0,0) * newCameraMatrix(1,1));
    newCameraMatrix(0,0) = newCameraMatrix(1,1) = focal;
    // We want the principal point at the image center in the resulting images.
    //newCameraMatrix(0,2) = imageSize.width/2;
    //newCameraMatrix(1,2) = imageSize.height/2;
    cv::Mat tmp_dummy;
    cv::Mat rectification_3x3;
    if (calib.rectification.empty()) {
        calib.rectification = cv::Mat_<double>(3, 1, 0.0);
    }
    if (calib.rectification.size() != Size(3,3)) {
        cv::Rodrigues(calib.rectification, rectification_3x3);
    }

    cv::initUndistortRectifyMap(calib.cameraMatrix,
                                calib.distCoeffs,
                                rectification_3x3,
                                newCameraMatrix,
                                imageSize,
                                CV_32FC2,
                                calib.undistortRectifyMap,
                                tmp_dummy);

    clog::L(__func__, 2) << "final cameraMatrix: " << newCameraMatrix << std::endl;
    clog::L(__func__, 2) << "arguments for pointcloud renderer: -f " << std::sqrt(newCameraMatrix(0,0) * newCameraMatrix(1,1))
                         << " --cx " << newCameraMatrix(0,2) << " --cy " << newCameraMatrix(1,2) << std::endl;

    return calib.undistortRectifyMap;
}

Mat Calib::getCachedUndistortRectifyMap(std::string const& calibName) {
    CalibResult & calib = getCalib(calibName);
    if (calib.undistortRectifyMap.size() != imageSize) {
        calculateUndistortRectifyMap(calib);
    }
    return calib.undistortRectifyMap;
}


bool Calib::hasCalibName(const string &name) const {
    auto it = calibrations.find(name);
    if (it == calibrations.end()) {
        return false;
    }
    CalibResult const& c = it->second;
    if (c.imageFiles != imageFiles) {
        return false;
    }
    if (c.rvecs.size() != imageFiles.size()) {
        return false;
    }
    if (c.tvecs.size() != imageFiles.size()) {
        return false;
    }
    return true;
}

void Calib::normalizeRotationVector(Mat &vector) {
    cv::Mat mat(3,3, vector.type());
    cv::Rodrigues(vector, mat);
    cv::Rodrigues(mat, vector);
}

void Calib::normalizeRotationVector(double vector[]) {
    cv::Mat_<double> vec(3,1);
    for (size_t ii = 0; ii < 3; ++ii) {
        vec(int(ii)) = vector[ii];
    }
    normalizeRotationVector(vec);
    for (size_t ii = 0; ii < 3; ++ii) {
        vector[ii] = vec(int(ii));
    }
}

double Calib::getMarkerSize() const {
    return markerSize;
}

void Calib::invalidateCache() {
    preparedCalib = false;
    preparedOpenCVCalib = false;
    imagePoints.clear();
    objectPoints.clear();
}

int Calib::getCornerIdFactor() const {
    return cornerIdFactor;
}

int Calib::computeCornerIdFactor(const int recursion_depth) {
    int result = 1;
    if (recursion_depth > 0) {
        result = 10;
        for (int ii = 1; ii < recursion_depth; ++ii) {
            result *= 5;
        }
    }
    return result;
}

void Calib::setValidPages(const std::vector<int> &_pages) {
    validPages = _pages;
}

void Calib::purgeInvalidPages() {
    invalidateCache();
    runningstats::RunningStats percent_stats;
    for (auto& it : data) {
        std::vector<hdmarker::Corner> cleaned = purgeInvalidPages(it.second.getCorners(), validPages);
        if (cleaned.size() < it.second.size()) {
            clog::L(__func__, 2) << "In image " << it.first << " removed " << it.second.size() - cleaned.size() << " out of " << it.second.size() << " corners" << std::endl;
            double const percentage = it.second.size() > 0 ? 100.0*double(it.second.size() - cleaned.size())/it.second.size() : 100;
            percent_stats.push_unsafe(percentage);
            it.second.replaceCorners(cleaned);
        }
    }
    clog::L(__func__, 1) << percent_stats.print();
}

std::vector<Corner> Calib::purgeInvalidPages(const std::vector<Corner> &in, const std::vector<int> &valid_pages) {
    std::vector<hdmarker::Corner> result;
    result.reserve(in.size());
    for (const auto& c : in) {
        if (isValidPage(c.page, valid_pages)) {
            result.push_back(c);
        }
    }
    return result;
}

bool Calib::isValidPage(const int page, const std::vector<int> &valid_pages) {
    for (const auto valid : valid_pages) {
        if (page == valid) {
            return true;
        }
    }
    return false;
}

bool Calib::isValidPage(const int page) const {
    return isValidPage(page, validPages);
}

bool Calib::isValidPage(const Corner &c) const {
    return isValidPage(c.page);
}

void Calib::setRecursionDepth(int const _recursionDepth) {
    recursionDepth = _recursionDepth;
    cornerIdFactor = computeCornerIdFactor(recursionDepth);
}

void Calib::scaleCornerIds(std::vector<Corner> &corners, int factor) {
    for (hdmarker::Corner& c: corners) {
        c.id *= factor;
    }
}

void Calib::prepareOpenCVCalibration() {
    if (preparedOpenCVCalib && imagePoints.size() == data.size() && objectPoints.size() == data.size()) {
        return;
    }
    preparedOpenCVCalib = true;
    preparedCalib = false;

    imagePoints = std::vector<std::vector<cv::Point2f> >(data.size());
    objectPoints = std::vector<std::vector<cv::Point3f> >(data.size());
    imageFiles.resize(data.size());

    size_t ii = 0;
    for (std::pair<const std::string, CornerStore> const& it : data) {
        it.second.getMajorPoints(imagePoints[ii], objectPoints[ii], *this);
        imageFiles[ii] = it.first;
        ++ii;
    }
}

bool Calib::hasFile(const string filename) const {
    return data.end() != data.find(filename);
}

size_t Calib::getId(const string &filename) const {
    for (size_t ii = 0; ii < imageFiles.size(); ++ii) {
        if (filename == imageFiles[ii]) {
            return ii;
        }
    }
    throw std::runtime_error("Image not found");
}

void Calib::only_green(bool only_green) {
    useOnlyGreen = only_green;
}

std::vector<double> Calib::mat2vec(const Mat &in) {
    std::vector<double> result;
    result.reserve(size_t(in.cols*in.rows+1));
    cv::Mat_<double> _in(in);
    for (auto const& it : _in) {
        result.push_back(it);
    }
    return result;
}

cv::Mat_<double> Calib::vec2mat(const std::vector<double> &in) {
    cv::Mat_<double> result;
    for (auto const& it : in) {
        result.push_back(it);
    }
    return result;
}

std::vector<double> Calib::point2vec3f(const Point3f &in) {
    return {double(in.x), double(in.y), double(in.z)};
}

Point3f Calib::vec2point3f(const std::vector<double> &in) {
    if (in.size() < 3) {
        throw std::runtime_error("Less than 3 elements in input vector.");
    }
    return cv::Point3f(float(in[0]), float(in[1]), float(in[2]));
}

Point3f Calib::getInitial3DCoord(const Corner &c, const double z) const {
    return getInitial3DCoord(getSimpleId(c), z);
}

Point3f Calib::getInitial3DCoord(const Point3i &c, const double z) const {
    cv::Point3f res(float(c.x), float(c.y), float(z));
    switch (c.z) {
    case 1: res.x += cornerIdFactor * 32; break;
    case 2: res.x += cornerIdFactor * 64; break;

    case 3: res.y += cornerIdFactor * 32; break;
    case 4: res.y += cornerIdFactor * 32; res.x += cornerIdFactor * 32; break;
    case 5: res.y += cornerIdFactor * 32; res.x += cornerIdFactor * 64; break;

    case 7: res.x += cornerIdFactor * 32; break;
    }

    res.x *= float(markerSize / cornerIdFactor);
    res.y *= float(markerSize / cornerIdFactor);

    return res;
}

void Calib::setMarkerSize(double const size) {
    if (size != markerSize) {
        double const ratio = size/markerSize;
        clog::L("Calib::setMarkerSize", 2) << "Setting new marker size, scaling translation vectors, ratio: " << ratio;
        for (std::pair<const std::string, CalibResult> & it : calibrations) {
            it.second.scaleResult(ratio);
        }
        markerSize = size;
    }
}

CornerStore Calib::get(const string filename) const {
    auto const it = data.find(filename);
    if (it != data.end()) {
        return it->second;
    }
    throw std::runtime_error(std::string("File ") + filename + " not found in data.");
}

CornerStore Calib::getUnion() const {
    CornerStore res;
    for (const auto& it : data) {
        for (size_t ii = 0; ii < it.second.size(); ++ii) {
            res.push_conditional(it.second.get(ii));
        }
    }
    return res;
}

std::vector<hdmarker::Corner> Calib::readCorners(
        const std::string& input_file,
        int &width,
        int &height) {
    std::vector<hdmarker::Corner> corners;
    FileStorage pointcache(input_file, FileStorage::READ);
    FileNode n = pointcache["corners"]; // Read string sequence - Get node
    if (n.type() != FileNode::SEQ) {
        throw std::runtime_error("Corners is not a sequence! FAIL");
    }

    FileNodeIterator it = n.begin(), it_end = n.end(); // Go through the node
    for (; it != it_end; ++it) {
        hdmarker::Corner c;
        *it >> c;
        corners.push_back(c);
    }
    if (pointcache["imageWidth"].isInt() && pointcache["imageHeight"].isInt()) {
        pointcache["imageWidth"] >> width;
        pointcache["imageHeight"] >> height;
    }
    return corners;
}

std::vector<Corner> Calib::readCorners(const string &input_file) {
    int width = 0, height = 0;
    return readCorners(input_file, width, height);
}

vector<Corner> Calib::getMainMarkers(const std::string input_file,
                                     const float effort,
                                     const bool demosaic,
                                     const bool raw) {
    std::vector<Corner> result;

    this->effort = effort;
    this->demosaic = demosaic;
    this->libraw = raw;

    std::string const hdm_gz_cache_file = input_file + "-pointcache.hdmarker.gz";
    if (fs::is_regular_file(hdm_gz_cache_file)) {
        Corner::readFile(hdm_gz_cache_file, result);
        if (!validPages.empty()) {
            result = purgeInvalidPages(result, validPages);
        }
        result = filter_duplicate_markers(result);
        return result;
    }

    std::string const cv_gz_cache_file = input_file + "-pointcache.yaml.gz";
    if (fs::is_regular_file(cv_gz_cache_file)) {
        Corner::readFile(cv_gz_cache_file, result);
        int width = 0, height = 0;
        result = readCorners(cv_gz_cache_file, width, height);
        if (0 != width && 0 != height) {
            imageSize.width = width;
            imageSize.height = height;
            resolutionKnown = true;
        }
        if (!validPages.empty()) {
            result = purgeInvalidPages(result, validPages);
        }
        result = filter_duplicate_markers(result);
        Corner::writeGzipFile(hdm_gz_cache_file, result);
        return result;
    }

    cv::Mat img_scaled = getImageScaled(input_file);

    Marker::init();
    detect(img_scaled, result, use_rgb, 0, 10, effort);
    std::map<int, int> counter;
    for (auto const& it : result) {
        counter[it.page]++;
    }
    clog::L(__func__, 2) << "Detected page stats: " << std::endl;
    for (auto const& it : counter) {
        clog::L(__func__, 2) << it.first << ": " << it.second << std::endl;
    }

    if (!validPages.empty()) {
        result = purgeInvalidPages(result, validPages);
    }
    result = filter_duplicate_markers(result);

    Corner::writeGzipFile(hdm_gz_cache_file, result);

    return result;
}

cv::Mat Calib::getImageScaled(std::string const& input_file) {
    cv::Mat img = readImage(input_file, demosaic, libraw, useOnlyGreen);
    if (img.empty()) {
        clog::L("getCorners", 1) << "Input file empty, aborting." << std::endl;
        return {};
    }
    setImageSize(img);

    return scaleImage(img);
}

vector<Corner> Calib::getSubMarkers(const std::string input_file,
                                    const float effort,
                                    const bool demosaic,
                                    const bool raw,
                                    bool * is_clean) {
    std::vector<Corner> result;

    this->effort = effort;
    this->demosaic = demosaic;
    this->libraw = raw;

    std::string const hdm_clean_gz_cache_file = input_file + "-submarkers-clean.hdmarker.gz";
    if (fs::is_regular_file(hdm_clean_gz_cache_file)) {
        Corner::readFile(hdm_clean_gz_cache_file, result);
        if (nullptr != is_clean) {
            *is_clean = true;
        }
        return result;
    }
    if (nullptr != is_clean) {
        *is_clean = false;
    }

    std::string const hdm_gz_cache_file = input_file + "-submarkers.hdmarker.gz";
    if (fs::is_regular_file(hdm_gz_cache_file)) {
        Corner::readFile(hdm_gz_cache_file, result);
        return result;
    }

    std::string const cv_gz_cache_file = input_file + "-submarkers.yaml.gz";
    if (fs::is_regular_file(cv_gz_cache_file)) {
        Corner::readFile(cv_gz_cache_file, result);
        int width = 0, height = 0;
        result = readCorners(cv_gz_cache_file, width, height);
        if (0 != width && 0 != height) {
            imageSize.width = width;
            imageSize.height = height;
            resolutionKnown = true;
        }
        Corner::writeGzipFile(hdm_gz_cache_file, result);
        return result;
    }

    std::vector<Corner> main_markers = getMainMarkers(input_file, effort, demosaic, raw);

    cv::Mat img = getImageScaled(input_file);

    double msize = 1.0;
    refineRecursiveByPage(img, main_markers, result, recursionDepth, msize);
    clog::L(__func__, 1) << "Number of detected submarkers: " << result.size() << std::endl;
    if (!validPages.empty()) {
        result = purgeInvalidPages(result, validPages);
    }
    clog::L(__func__, 1) << "Number of detected submarkers after purgeInvalidPages: " << result.size() << std::endl;

    if (result.size() <= main_markers.size()) {
        clog::L(__func__, 0) << "Warning: Number of submarkers (" << std::to_string(result.size())
                             << ") smaller than or equal to the number of corners (" << std::to_string(main_markers.size()) << "), in input file"
                             << input_file << ", scaling ids." << std::endl;
        int factor = 10;
        for (int ii = 1; ii < recursionDepth; ++ii) {
            factor *=5;
        }
        result = main_markers;
        for (auto& it : result) {
            it.id *= factor;
        }
    }

    Corner::writeGzipFile(hdm_gz_cache_file, result);

    return result;
}

cv::Mat Calib::scaleImage(cv::Mat const& img) {
    cv::Mat img_scaled = img.clone();
    double min = 0;
    double max = 0;
    cv::minMaxIdx(img_scaled, &min, &max);
    if (img_scaled.depth() == CV_16U) {
        img_scaled *= std::floor(65535.0 / max);
        img_scaled.convertTo(img_scaled, CV_8UC1, 1.0 / 256.0);
    }
    else if (img_scaled.depth() == CV_8U) {
        img_scaled *= std::floor(255.0 / max);
    }
    if (img_scaled.channels() == 1 && (img_scaled.depth() == CV_16U || img_scaled.depth() == CV_16S)) {
        clog::L(__func__, 0) << "Input image is 1 channel, 16 bit, converting for painting to 8 bit." << std::endl;
        img_scaled.convertTo(img_scaled, CV_8UC1, 1.0 / 256.0);
    }
    return img_scaled;
}

vector<Corner> Calib::getCorners(const std::string input_file,
                                 const float effort,
                                 const bool demosaic,
                                 const bool raw) {
    vector<Corner> corners, submarkers;

    this->effort = effort;
    this->demosaic = demosaic;
    this->libraw = raw;

    Mat img, paint;

    corners = getMainMarkers(input_file, effort, demosaic, raw);
    if (recursionDepth > 0) {
        submarkers = getSubMarkers(input_file, effort, demosaic, raw);
    }
    if (!resolutionKnown || 0 == imageSize.width || 0 == imageSize.height) {
        if (raw) {
            imageSize = read_raw_imagesize(input_file);
        }
        else {
            img = cv::imread(input_file, cv::IMREAD_GRAYSCALE);
            setImageSize(img);
        }
        resolutionKnown = true;
    }
    cv::Mat img_scaled;

    if (plotMarkers) {
        img_scaled = getImageScaled(input_file);
        img = readImage(input_file, demosaic, libraw, useOnlyGreen);
        if (img.empty()) {
            clog::L("getCorners", 1) << "Input file empty, aborting." << std::endl;
            return {};
        }
        if (demosaic && !fs::is_regular_file(input_file + "-demosaiced.png")) {
            cv::imwrite(input_file + "-demosaiced.png", img_scaled);
        }
        setImageSize(img);
        paint = img.clone();
    }
    if (plotMarkers && paint.channels() == 1) {
        cv::Mat tmp[3] = {paint, paint, paint};
        cv::merge(tmp, 3, paint);
    }
    if (paint.depth() == CV_16U || paint.depth() == CV_16S) {
        clog::L(__func__, 1) << "Input image 16 bit, converting for painting to 8 bit." << std::endl;
        paint.convertTo(paint, CV_8UC3, 1.0 / 256.0);
    }
    //clog::L(__func__, 0) << "Paint type: " << paint.type() << std::endl;
    //clog::L(__func__, 0) << "Paint depth: " << paint.depth() << std::endl;
    //clog::L(__func__, 0) << "Input image size of file " << input_file << ": " << img.size << std::endl;

    Marker::init();

    if (img.depth() == CV_16U || img.depth() == CV_16S) {
        clog::L(__func__, 0) << "Input image for marker detection 16 bit, converting for painting to 8 bit." << std::endl;
        paint.convertTo(paint, CV_8UC1, 1.0 / 256.0);
    }

    Mat gray;
    if (img_scaled.channels() != 1) {
        cvtColor(img_scaled, gray, cv::COLOR_BGR2GRAY);
    }
    else {
        gray = img_scaled;
    }

    if (recursionDepth > 0) {
        clog::L(__func__, 0) << "Drawing sub-markers" << std::endl;
        cv::Mat_<uint8_t> main_markers_area = getMainMarkersArea(submarkers);
        if (plotMarkers) {
            cv::Mat paint2 = paint.clone();
            for (hdmarker::Corner const& c : submarkers) {
                circle(paint2, c.p, 3, Scalar(0,0,255,0), -1, LINE_AA);
            }
            if (!fs::is_regular_file(input_file + "-2.png")) {
                imwrite(input_file + "-2.png", paint2);
            }
            if (!fs::is_regular_file(input_file + "-main-area.png")) {
                cv::imwrite(input_file + "-main-area.png", main_markers_area);
            }
        }
    }

    if (plotMarkers) {
        //std::sort(submarkers.begin(), submarkers.end(), CornerIdSort());
        cv::Mat paint_submarkers = paint.clone();
        if (recursionDepth > 0 && plotSubMarkers) {
            int paint_size_factor = 2;
            if (paint.cols < 3000 && paint.rows < 3000) {
                paint_size_factor = 5;
            }
            cv::resize(paint_submarkers, paint_submarkers, cv::Size(), paint_size_factor, paint_size_factor, cv::INTER_NEAREST);
            paintSubmarkers(submarkers, paint_submarkers, paint_size_factor);
            imwrite(input_file + "-sub.png", paint_submarkers);
        }
        for(size_t ii = 0; ii < corners.size(); ++ii) {
            Corner const& c = corners[ii];

            Point2f p1, p2;
            cv::Scalar const& font_color = color_circle[ii % color_circle.size()];

            std::string const text = to_string(c.id.x) + "/" + to_string(c.id.y) + "/" + to_string(c.page);
            circle(paint, c.p, 1, Scalar(0,0,0,0), 2);
            circle(paint, c.p, 1, Scalar(0,255,0,0));
            putText(paint, text.c_str(), c.p, FONT_HERSHEY_PLAIN, 1.2, Scalar(0,0,0,0), 2, cv::LINE_AA);
            putText(paint, text.c_str(), c.p, FONT_HERSHEY_PLAIN, 1.2, font_color, 1, cv::LINE_AA);

            std::string const text_page = to_string(c.page);
            double font_size = 2;
            cv::Point2f const point_page = c.p + (c.size/2 - float(font_size*5))*cv::Point2f(1,1);
            putText(paint, text_page.c_str(), point_page, FONT_HERSHEY_PLAIN, font_size, Scalar(0,0,0,0), 2, cv::LINE_AA);
            putText(paint, text_page.c_str(), point_page, FONT_HERSHEY_PLAIN, font_size, color_circle[size_t(c.page) % color_circle.size()], 1, cv::LINE_AA);
        }
        imwrite(input_file + "-1.png", paint);
    }


    if (recursionDepth > 0) {
        return submarkers;
    }

    return corners;

}

Mat Calib::readImage(std::string const& input_file,
                     bool const demosaic,
                     bool const raw,
                     bool const useOnlyGreen) {
    cv::Mat img;
    if (demosaic) {
        if (raw) {
            img = read_raw(input_file);
        }
        else {
            img = cv::imread(input_file, cv::IMREAD_GRAYSCALE);
        }
        double min_val = 0, max_val = 0;
        cv::minMaxIdx(img, &min_val, &max_val);
        //clog::L(__func__, 2) << "Image min/max: " << min_val << " / " << max_val << std::endl;

        cvtColor(img, img, COLOR_BayerBG2BGR); // RG BG GB GR
    }
    else {
        img = cv::imread(input_file);
        //clog::L(__func__, 2) << "Input file " << input_file << " image size: " << img.size() << std::endl;
    }
    if (useOnlyGreen) {
        if (img.channels() > 1) {
            cv::Mat split[3];
            cv::split(img, split);
            img = split[1];
        }
    }
    return img;
}

std::vector<Corner> filter_duplicate_markers(const std::vector<Corner> &in) {
    std::vector<hdmarker::Corner> result;
    result.reserve(in.size());
    for (size_t ii = 0; ii < in.size(); ++ii) {
        bool has_duplicate = false;
        const hdmarker::Corner& a = in[ii];
        if (a.id.x == 32 || a.id.y == 32) {
            for (size_t jj = 0; jj < in.size(); ++jj) {
                const hdmarker::Corner& b = in[jj];
                if (cv::norm(a.p-b.p) < double(a.size + b.size)/20) {
                    has_duplicate = true;
                    break;
                }
            }
        }
        if (!has_duplicate) {
            result.push_back(a);
        }
    }
    return result;
}

double Calib::openCVCalib(bool const simple) {
    prepareOpenCVCalibration();

    std::string const name = simple ? "SimpleOpenCV" : "OpenCV";
    bool has_precalib = false;
    if (simple) {
        has_precalib = hasCalibName("SimpleOpenCV");
    }
    else {
        if (!hasCalibName("OpenCV") && hasCalibName("SimpleOpenCV")) {
            calibrations["OpenCV"] = calibrations["SimpleOpenCV"];
        }
    }
    CalibResult & res = calibrations[name];

    int flags = 0;
    flags |= CALIB_USE_LU;
    if (has_precalib) {
        flags |= CALIB_USE_INTRINSIC_GUESS;
        flags |= CALIB_USE_EXTRINSIC_GUESS;
    }
    if (simple) {
        flags |= CALIB_FIX_PRINCIPAL_POINT;
        flags |= CALIB_FIX_ASPECT_RATIO;
        flags |= CALIB_ZERO_TANGENT_DIST;
        flags |= CALIB_FIX_K1 | CALIB_FIX_K2 | CALIB_FIX_K3 | CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_K6;
        flags |= CALIB_FIX_S1_S2_S3_S4;
        flags |= CALIB_FIX_TAUX_TAUY;
    }
    if (!simple) {
        flags |= CALIB_RATIONAL_MODEL;
        flags |= CALIB_THIN_PRISM_MODEL;
        flags |= CALIB_TILTED_MODEL;
    }
    if (imageSize.height == 5320 && imageSize.width == 7968) { // Hacky detection of my Sony setup.
        res.cameraMatrix = (Mat_<double>(3,3) << 12937, 0, 4083, 0, 12978, 2636, 0, 0, 1);
        flags |= CALIB_USE_INTRINSIC_GUESS;
    }
    clog::L(__func__, 1) << "Initial camera matrix: " << std::endl << res.cameraMatrix << std::endl;

    double result_err = cv::calibrateCamera (
                objectPoints,
                imagePoints,
                imageSize,
                res.cameraMatrix,
                res.distCoeffs,
                res.rvecs,
                res.tvecs,
                res.stdDevIntrinsics,
                res.stdDevExtrinsics,
                res.perViewErrors,
                flags
                );

    clog::L(__func__, 1) << "RMSE: " << result_err << std::endl
                         << "Camera Matrix: " << std::endl << res.cameraMatrix << std::endl;
    //<< "distCoeffs: " << std::endl << res.distCoeffs << std::endl;

    /*
    clog::L(__func__, 2) << "stdDevIntrinsics: " << std::endl << res.stdDevIntrinsics << std::endl
                         << "stdDevExtrinsics: " << std::endl << res.stdDevExtrinsics << std::endl
                         << "perViewErrors: " << std::endl << res.perViewErrors << std::endl;

    clog::L(__func__, 2) << "rvecs: " << std::endl;
    for (auto const& rvec: res.rvecs) {
        clog::L(__func__, 2) << rvec << std::endl;
    }

    clog::L(__func__, 2) << "tvecs: " << std::endl;
    for (auto const& tvec: res.tvecs) {
        clog::L(__func__, 2) << tvec << std::endl;
    }
    */

    cv::calibrationMatrixValues (
                res.cameraMatrix,
                imageSize,
                apertureWidth,
                apertureHeight,
                fovx,
                fovy,
                focalLength,
                principalPoint,
                aspectRatio
                );

    double const pixel_size = apertureWidth / imageSize.width;
    clog::L(__func__, 1) << "calibrationMatrixValues: " << std::endl
                         << "fovx: " << fovx << std::endl
                         << "fovy: " << fovy << std::endl
                         << "focalLength: " << focalLength << std::endl
                         << "principalPoint: " << principalPoint << std::endl
                         << "aspectRatio: " << aspectRatio << std::endl
                         << "input image size: " << imageSize << std::endl
                         << "pixel size (um): " << pixel_size * 1000 << std::endl << std::endl;

    cv::Point2d principal_point_offset = principalPoint - cv::Point2d(apertureWidth/2, apertureHeight/2);
    clog::L(__func__, 1) << "principal point offset: " << principal_point_offset << "mm; ~" << principal_point_offset/pixel_size << "px" << std::endl;

    clog::L(__func__, 1) << "focal length factor: " << res.cameraMatrix(0,0) / focalLength << std::endl;

    hasCalibration = true;

    res.imageFiles = imageFiles;

    return result_err;
}

double Calib::runCalib(const string name, const double outlier_threshold) {
    if ("OpenCV" == name) {
        return openCVCalib(false);
    }
    if ("SimpleOpenCV" == name) {
        return openCVCalib(true);
    }
    if ("SimpleCeres" == name) {
        return SimpleCeresCalib(outlier_threshold);
    }
    if ("Ceres" == name) {
        return CeresCalib(outlier_threshold);
    }
    if ("Flexible" == name) {
        return CeresCalibFlexibleTarget(outlier_threshold);
    }
    throw std::runtime_error(std::string("Calib type ") + name + " unknown");
}

void Calib::setPlotMarkers(bool plot) {
    plotMarkers = plot;
}

void Calib::setPlotSubMarkers(bool plot) {
    plotSubMarkers = plot;
}

void Calib::setImageSize(const Mat &img) {
    imageSize = cv::Size(img.size());
    resolutionKnown = true;
}

Point2f Calib::meanResidual(const std::vector<std::pair<Point2f, Point2f> > &data) {
    cv::Point2f sum(0,0);
    for (auto const& it : data) {
        sum += it.first - it.second;
    }
    return cv::Point2f(sum.x/data.size(), sum.y/data.size());
}

Point3i Calib::getSimpleId(const Corner &marker) {
    return cv::Point3i(marker.id.x, marker.id.y, marker.page);
}

void Calib::findOutliers(
        std::string const& calib_name,
        const double threshold,
        const size_t image_index,
        std::vector<hdmarker::Corner> & outliers) {
    std::vector<cv::Point2d> markers, projections;
    getReprojections(calibrations[calib_name], image_index, markers, projections);

    runningstats::RunningStats inlier_stats, outlier_stats;
    for (size_t ii = 0; ii < markers.size(); ++ii) {
        double const error = distance(markers[ii], projections[ii]);
        if (error < threshold) {
            inlier_stats.push_unsafe(error);
            continue;
        }
        outlier_stats.push_unsafe(error);
        hdmarker::Corner c = data[imageFiles[image_index]].get(ii);
        outliers.push_back(c);
        /*
        clog::L(__func__, 3) << "found outlier in image " << imageFiles[image_index]
                                << ": id " << c.id << ", " << c.page << ", marker: "
                                << markers[ii] << ", proj: " << projections[ii]
                                   << ", dist: " << error << std::endl;
                                   */
    }
    clog::L(__func__, 2) << "Stats for " << imageFiles[image_index] << ": inliers: " << inlier_stats.print() << ", outliers: " << outlier_stats.print() << std::endl;
}

Point2f Calib::project(cv::Mat_<double> const& cameraMatrix, const Vec3d &point) const {
    double const p[3] = {point[0], point[1], point[2]};
    double result[2] = {0,0};
    double focal[2] = {cameraMatrix(0,0), cameraMatrix(1,1)};
    double principal[2] = {cameraMatrix(0,2), cameraMatrix(1,2)};
    double const  R[9] = {1,0,0,   0,1,0,   0,0,1};
    double const t[3] = {0,0,0};
    project(p, result, focal, principal, R, t);
    return cv::Point2f(float(result[0]), float(result[1]));
}

Vec3d Calib::get3DPoint(CalibResult& calib, const Corner &c, const Mat &_rvec, const Mat &_tvec) {
    cv::Mat_<double> rvec(_rvec);
    cv::Mat_<double> tvec(_tvec);
    cv::Point3f _src = getInitial3DCoord(c);
    _src += calib.objectPointCorrections[getSimpleId(c)];
    double src[3] = {double(_src.x), double(_src.y), double(_src.z)};
    double rot[9];
    double rvec_data[3] = {rvec(0), rvec(1), rvec(2)};
    double tvec_data[3] = {tvec(0), tvec(1), tvec(2)};
    rot_vec2mat(rvec_data, rot);
    double _result[3] = {0,0,0};
    get3DPoint(src, _result, rot, tvec_data);

    return cv::Vec3d(_result[0], _result[1], _result[2]);
}

Vec3d Calib::get3DPointWithoutCorrection(const Corner &c, const Mat &_rvec, const Mat &_tvec) {
    cv::Point3f _src = getInitial3DCoord(c);
    return get3DPointWithoutCorrection(_src, _rvec, _tvec);
}

Vec3d Calib::get3DPointWithoutCorrection(const cv::Point3f &_src, const Mat &_rvec, const Mat &_tvec) {
    cv::Mat_<double> rvec(_rvec);
    cv::Mat_<double> tvec(_tvec);
    double src[3] = {double(_src.x), double(_src.y), double(_src.z)};
    double rot[9];
    double rvec_data[3] = {rvec(0), rvec(1), rvec(2)};
    double tvec_data[3] = {tvec(0), tvec(1), tvec(2)};
    rot_vec2mat(rvec_data, rot);
    double _result[3] = {0,0,0};
    get3DPoint(src, _result, rot, tvec_data);

    return cv::Vec3d(_result[0], _result[1], _result[2]);
}

string Calib::tostringLZ(size_t num, size_t min_digits) {
    std::string result = std::to_string(num);
    if (result.size() < min_digits) {
        result = std::string(min_digits - result.size(), '0') + result;
    }
    return result;
}

template<class Point>
double Calib::distance(const Point a, const Point b) {
    Point residual = a-b;
    return std::sqrt(double(residual.dot(residual)));
}

//template double Calib::distance<cv::Point_<float> >(const cv::Point_<float>, const cv::Point_<float>);
template double Calib::distance<cv::Point_<float> >(cv::Point_<float>, cv::Point_<float>);
template double Calib::distance(const cv::Point_<double>, const cv::Point_<double>);

template<class C>
bool cmpSimpleIndex3<C>::operator()(const C &a, const C &b) const {
    if (a.z != b.z) return a.z < b.z;
    if (a.y != b.y) return a.y < b.y;
    return a.x < b.x;
}

template bool cmpSimpleIndex3<cv::Point3i>::operator()(const cv::Point3i &, const cv::Point3i &b) const;

bool cmpPoint3i::operator()(const cv::Point3i &a, const cv::Point3i &b) const {
    if (a.z != b.z) return a.z < b.z;
    if (a.y != b.y) return a.y < b.y;
    return a.x < b.x;
}


template<class F, class T>
void Calib::get3DPoint(const F p[], T result[], const T R[], const T t[]) {
    T const X(p[0]), Y(p[1]), Z(p[2]);
    T& x = result[0];
    T& y = result[1];
    T& z = result[2];
    z = R[6]*X + R[7]*Y + R[8]*Z + t[2];
    x = R[0]*X + R[1]*Y + R[2]*Z + t[0];
    y = R[3]*X + R[4]*Y + R[5]*Z + t[1];
}

template void Calib::get3DPoint(const double [], double [], const double [], const double []);

template<class Point>
bool Calib::validPixel(const Point &p, const Size &image_size) {
    return p.x >= 0 && p.y >= 0 && p.x+1 <= image_size.width && p.y+1 <= image_size.height;
}

template bool Calib::validPixel(const cv::Point2f&, const cv::Size&);



} // namespace hdcalib
