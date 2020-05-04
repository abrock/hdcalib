#include "hdcalib.h"

#include <ceres/ceres.h>
#include <opencv2/optflow.hpp>
#include <opencv2/video/tracking.hpp>
#include <runningstats/runningstats.h>
#include <catlogger/catlogger.h>
#include "gnuplot-iostream.h"


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
    clog::L(__func__, 2) << "Outlier percentage by image:" << std::endl;
    runningstats::RunningStats percent_stats;
    for (auto& it : data) {
        size_t const before = it.second.size();
        it.second.difference(subtrahend);
        size_t const after = it.second.size();
        double const percent = (double(before - after)/before)*100.0;
        percent_stats.push(percent);
        clog::L(__func__, 2) << it.first << ": removed " << (before-after) << " out of " << before << " corners (" << percent << "%)" << std::endl;
    }
    clog::L(__func__, 1) << "Removeal percentage stats: " << percent_stats.print() << std::endl;
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
        cv::Point3f current_objPoint = objPoints[jj] + calib.objectPointCorrections[getSimpleId(store.get(jj))];

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
    return result;
}

void Calib::exportPointClouds(const string &calib_name) {
    if (!hasCalibName(calib_name)) {
        throw std::runtime_error(std::string("Calib name ") + calib_name + " not available in exportPointClouds.");
    }
    CalibResult & calib = calibrations[calib_name];

    for (size_t ii = 0; ii < imageFiles.size(); ++ii) {
        std::string const& filename = imageFiles[ii];
        std::ofstream out(filename + "-target-points.txt", std::ofstream::out);
        CornerStore const& store = data[filename];
        for (size_t jj = 0; jj < store.size(); ++jj) {
            cv::Vec3d p = get3DPoint(calib, store.get(jj), calib.rvecs[ii], calib.tvecs[ii]);
            out << p[0] << "\t" << p[1] << "\t" << p[2] << std::endl;
        }
    }
}

Mat Calib::calculateUndistortRectifyMap(CalibResult & calib) {
    cv::Mat_<double> newCameraMatrix = calib.cameraMatrix.clone();
    // We want the principal point at the image center in the resulting images.
    newCameraMatrix(0,2) = imageSize.width/2;
    newCameraMatrix(1,2) = imageSize.height/2;
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

    return calib.undistortRectifyMap;
}

Mat Calib::getCachedUndistortRectifyMap(std::string const& calibName) {
    CalibResult & calib = calibrations[calibName];
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
    result.reserve(size_t(in.cols*in.rows));
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
    markerSize = size;
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

vector<Corner> Calib::getCorners(const std::string input_file,
                                 const float effort,
                                 const bool demosaic,
                                 const bool raw) {
    std::string pointcache_file = input_file + "-pointcache.yaml.gz";
    std::string submarkers_file = input_file + "-submarkers.yaml.gz";
    vector<Corner> corners, submarkers;

    Mat img, paint;

    bool read_cache_success = false;
    try {
        if (fs::exists(pointcache_file)) {
            int width = 0, height = 0;
            corners = readCorners(pointcache_file, width, height);
            if (0 != width && 0 != height) {
                imageSize.width = width;
                imageSize.height = height;
                resolutionKnown = true;
            }
            read_cache_success = true;
        }
    }
    catch (const Exception& e) {
        clog::L(__func__, 0) << "Reading pointcache file failed with exception: " << std::endl
                             << e.what() << std::endl;
        read_cache_success = false;
    }
    if (read_cache_success) {
        if (!validPages.empty()) {
            corners = purgeInvalidPages(corners, validPages);
        }
        corners = filter_duplicate_markers(corners);
        clog::L(__func__, 2) << "Got " << corners.size() << " corners from pointcache file" << std::endl;
    }

    bool read_submarkers_success = false;
    try {
        if (fs::exists(submarkers_file)) {
            int width = 0, height = 0;
            submarkers = readCorners(submarkers_file, width, height);
            if (0 != width && 0 != height) {
                imageSize.width = width;
                imageSize.height = height;
                resolutionKnown = true;
            }
            if (submarkers.size() > corners.size()) {
                read_submarkers_success = true;
            }
        }
    }
    catch (const Exception& e) {
        clog::L(__func__, 0) << "Reading pointcache file failed with exception: " << std::endl
                             << e.what() << std::endl;
        read_submarkers_success = false;
    }
    if (read_submarkers_success) {
        if (!validPages.empty()) {
            submarkers = purgeInvalidPages(submarkers, validPages);
        }
        submarkers = filter_duplicate_markers(submarkers);
        clog::L(__func__, 2) << "Got " << submarkers.size() << " submarkers from submarkers file" << std::endl;
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

    if (plotMarkers || !read_cache_success || (recursionDepth > 0 && !read_submarkers_success)) {
        img = readImage(input_file, demosaic, raw, useOnlyGreen);
        if (img.empty()) {
            clog::L("getCorners", 1) << "Input file empty, aborting." << std::endl;
            return {};
        }
        if (demosaic) {
            cv::imwrite(input_file + "-demosaiced.png", img);
        }
        img_scaled = img.clone();
        setImageSize(img);

        double min = 0;
        double max = 0;
        cv::minMaxIdx(img, &min, &max);
        if (img.depth() == CV_16U) {
            img_scaled *= std::floor(65535.0 / max);
            img_scaled.convertTo(img_scaled, CV_8UC1, 1.0 / 256.0);
        }
        else if (img.depth() == CV_8U) {
            img_scaled *= std::floor(255.0 / max);
        }
        if (img.channels() == 1 && (img.depth() == CV_16U || img.depth() == CV_16S)) {
            clog::L(__func__, 0) << "Input image is 1 channel, 16 bit, converting for painting to 8 bit." << std::endl;
            img.convertTo(img, CV_8UC1, 1.0 / 256.0);
        }
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
    clog::L(__func__, 0) << "Paint type: " << paint.type() << std::endl;
    clog::L(__func__, 0) << "Paint depth: " << paint.depth() << std::endl;
    clog::L(__func__, 0) << "Input image size of file " << input_file << ": " << img.size << std::endl;

    Marker::init();

    if (img.depth() == CV_16U || img.depth() == CV_16S) {
        clog::L(__func__, 0) << "Input image for marker detection 16 bit, converting for painting to 8 bit." << std::endl;
        paint.convertTo(paint, CV_8UC1, 1.0 / 256.0);
    }
    if (!read_cache_success) {
        detect(img_scaled, corners, use_rgb, 0, 10, effort);
        std::map<int, int> counter;
        for (auto const& it : corners) {
            counter[it.page]++;
        }
        clog::L(__func__, 2) << "Detected page stats: " << std::endl;
        for (auto const& it : counter) {
            clog::L(__func__, 2) << it.first << ": " << it.second << std::endl;
        }

        if (!validPages.empty()) {
            corners = purgeInvalidPages(corners, validPages);
        }
        corners = filter_duplicate_markers(corners);

        FileStorage pointcache(pointcache_file, FileStorage::WRITE);
        pointcache << "corners" << "[";
        for (hdmarker::Corner const& c : corners) {
            pointcache << c;
        }
        pointcache << "]";
        pointcache << "imageWidth" << imageSize.width;
        pointcache << "imageHeight" << imageSize.height;
    }

    printf("final score %zu corners\n", corners.size());

    Mat gray;
    if (img_scaled.channels() != 1) {
        cvtColor(img_scaled, gray, cv::COLOR_BGR2GRAY);
    }
    else {
        gray = img_scaled;
    }

    if (recursionDepth > 0) {
        clog::L(__func__, 0) << "Drawing sub-markers" << std::endl;
        double msize = 1.0;
        if (!read_submarkers_success) {
            //hdmarker::refine_recursive(gray, corners, submarkers, recursionDepth, &msize);
            //piecewiseRefinement(gray, corners, submarkers, recursion_depth, msize);
            refineRecursiveByPage(gray, corners, submarkers, recursionDepth, msize);
            clog::L(__func__, 1) << "Number of detected submarkers: " << submarkers.size() << std::endl;
            if (!validPages.empty()) {
                submarkers = purgeInvalidPages(submarkers, validPages);
            }
            clog::L(__func__, 1) << "Number of detected submarkers after purgeInvalidPages: " << submarkers.size() << std::endl;
            FileStorage submarker_cache(submarkers_file, FileStorage::WRITE);
            submarker_cache << "corners" << "[";
            for (hdmarker::Corner const& c : submarkers) {
                submarker_cache << c;
            }
            submarker_cache << "]";
            submarker_cache.release();
        }
        cv::Mat_<uint8_t> main_markers_area = getMainMarkersArea(submarkers);
        std::vector<hdmarker::Corner> keep_submarkers;
        for (auto const& s : submarkers) {
            if (s.p.x >= 0 && s.p.y >= 0 && s.p.x < main_markers_area.cols && s.p.y < main_markers_area.rows) {
                if (main_markers_area(s.p) > 0) {
                    keep_submarkers.push_back(s);
                }
            }
        }
        submarkers = keep_submarkers;


        if (plotMarkers) {
            cv::Mat paint2 = paint.clone();
            for (hdmarker::Corner const& c : submarkers) {
                circle(paint2, c.p, 3, Scalar(0,0,255,0), -1, LINE_AA);
            }
            imwrite(input_file + "-2.png", paint2);
            cv::imwrite(input_file + "-main-area.png", main_markers_area);
        }

        if (submarkers.size() <= corners.size()) {
            clog::L(__func__, 0) << "Warning: Number of submarkers (" << std::to_string(submarkers.size())
                                 << ") smaller than or equal to the number of corners (" << std::to_string(corners.size()) << "), in input file"
                                 << input_file << ", scaling ids." << std::endl;
            int factor = 10;
            for (int ii = 1; ii < recursionDepth; ++ii) {
                factor *=5;
            }
            submarkers = corners;
            for (auto& it : submarkers) {
                it.id *= factor;
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
        clog::L(__func__, 2) << "Image min/max: " << min_val << " / " << max_val << std::endl;

        cvtColor(img, img, COLOR_BayerBG2BGR); // RG BG GB GR
    }
    else {
        img = cv::imread(input_file);
        clog::L(__func__, 2) << "Input file " << input_file << " image size: " << img.size() << std::endl;
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

    CalibResult & res = calibrations[simple ? "SimpleOpenCV" : "OpenCV"];

    int flags = 0;
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
        clog::L(__func__, 3) << "found outlier in image " << imageFiles[image_index]
                                << ": id " << c.id << ", " << c.page << ", marker: "
                                << markers[ii] << ", proj: " << projections[ii]
                                   << ", dist: " << error << std::endl;
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
    cv::Mat_<double> rvec(_rvec);
    cv::Mat_<double> tvec(_tvec);
    cv::Point3f _src = getInitial3DCoord(c);
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
