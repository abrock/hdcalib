#undef NDEBUG
#include <assert.h>   // reinclude the header to update the definition of assert()

#include "hdcalib.h"

#include <ceres/ceres.h>
#include <opencv2/optflow.hpp>
#include <opencv2/video/tracking.hpp>
#include <runningstats/runningstats.h>
#include <catlogger/catlogger.h>
#include "gnuplot-iostream.h"


#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

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

std::vector<std::vector<cv::Point2f> > Calib::getImagePoints() const
{
    return imagePoints;
}

std::vector<std::vector<cv::Point3f> > Calib::getObjectPoints() const
{
    return objectPoints;
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

void Calib::setCachePrefix(const string &val) {
    cache_prefix = val;
}

void Calib::saveReprojectionsJson(
        const string &filename,
        const std::vector<Vec2f> &features,
        const std::vector<Vec2f> &errors) const {
    namespace rs = rapidjson;
    rs::Document document_j;
    document_j.SetObject();
    rs::Document::AllocatorType& allocator = document_j.GetAllocator();
    document_j.AddMember("width", rs::Value().SetInt(imageSize.width), allocator);
    document_j.AddMember("height", rs::Value().SetInt(imageSize.height), allocator);

    document_j.AddMember("errors", rs::Value().SetArray(), allocator);
    document_j.AddMember("features", rs::Value().SetArray(), allocator);

    rs::Value errors_j(rs::kArrayType);
    for (size_t ii = 0; ii < errors.size(); ++ii) {
        rs::Value err;
        err.SetArray();
        err.PushBack(rs::Value().SetDouble(errors[ii][0]), allocator);
        err.PushBack(rs::Value().SetDouble(errors[ii][1]), allocator);
        document_j["errors"].PushBack(err, allocator);

        rs::Value feat;
        feat.SetArray();
        feat.PushBack(rs::Value().SetDouble(features[ii][0]), allocator);
        feat.PushBack(rs::Value().SetDouble(features[ii][1]), allocator);
        document_j["features"].PushBack(feat, allocator);
    }

    // 3. Stringify the DOM
    rs::StringBuffer buffer;
    rs::Writer<rs::StringBuffer> writer(buffer);
    document_j.Accept(writer);

    // Output
    std::ofstream out(filename);
    out << buffer.GetString() << std::endl;
}

void Calib::saveReprojectionsJsons(
        const string &prefix,
        const std::vector<Corner> &features,
        const std::vector<Vec2f> &errors) const {
    //       layer         color
    std::map<int, std::map<int, std::vector<cv::Vec2f> > > part_features, part_errors;
    std::map<int, std::vector<cv::Vec2f> > layer_features, layer_errors;
    std::map<int, std::map<int, runningstats::QuantileStats<float> > > part_stats;
    std::map<int, runningstats::QuantileStats<float> > layer_stats;
    runningstats::QuantileStats<float> all_stats;
    std::vector<cv::Vec2f> all_features;

    assert(features.size() == errors.size());
    for (size_t ii = 0; ii < errors.size(); ++ii) {
        int const layer = features[ii].layer;
        int const color = features[ii].color;
        all_features.push_back(features[ii].p);
        double const length = cv::norm(errors[ii]);
        all_stats.push_unsafe(length);
        part_stats[layer][color].push_unsafe(length);
        layer_stats[layer].push_unsafe(length);

        part_features[layer][color].push_back(cv::Vec2f(features[ii].p.x, features[ii].p.y));
        part_errors[layer][color].push_back(errors[ii]);

        layer_features[layer].push_back(cv::Vec2f(features[ii].p.x, features[ii].p.y));
        layer_errors[layer].push_back(errors[ii]);
    }

    saveReprojectionsJson(prefix + "-all.json", all_features, errors);
    clog::L("saveReprojections", 2) << "File: " << prefix << "-all.json median error: " << all_stats.getMedian() << " / " << all_stats.getCount();

    for (auto const& layer : part_features) {
        {
            std::string const filename = prefix + "-l" + std::to_string(layer.first) + ".json";
            saveReprojectionsJson(filename, layer_features[layer.first], layer_errors[layer.first]);
            clog::L("saveReprojections", 2) << "File: " << filename << " median error: " << layer_stats[layer.first].getMedian()
                                            << " / " << layer_stats[layer.first].getCount();
        }
        for (auto const& color : layer.second) {
            std::string const filename = prefix + "-l" + std::to_string(layer.first) + "-c" + std::to_string(color.first) + ".json";
            saveReprojectionsJson(filename, color.second, part_errors[layer.first][color.first]);
            clog::L("saveReprojections", 2) << "File: " << filename << " median error: " << part_stats[layer.first][color.first].getMedian()
                    << " / " << part_stats[layer.first][color.first].getCount();
        }
    }
}

void Calib::setIdRestriction(const int factor, const int mod) {
    restrict_factor_x = restrict_factor_y = factor;
    restrict_mod_x = restrict_mod_y = mod;
}

void Calib::setIdRestrictionX(const int factor, const int mod) {
    restrict_factor_x = factor;
    restrict_mod_x = mod;
}

void Calib::setIdRestrictionY(const int factor, const int mod) {
    restrict_factor_y = factor;
    restrict_mod_y = mod;
}

int Calib::getRestrictFactorX() const {
    return restrict_factor_x;
}

int Calib::getRestrictModX() const {
    return restrict_mod_x;
}

int Calib::getRestrictFactorY() const {
    return restrict_factor_y;
}

int Calib::getRestrictModY() const {
    return restrict_mod_y;
}



void Calib::useRobustMainMarkerDetection(bool val) {
    use_robust_main_marker_detection = val;
}

void Calib::setMarkerCountLimit(int limit) {
    if (markerCountLimit != limit) {
        markerCountLimit = limit;
        preparedCalib = preparedOpenCVCalib = preparedOpenCVROCalib = false;
    }
}

int Calib::getMarkerCountLimit() const {
    return markerCountLimit;
}

bool Calib::isSelected(const Corner &c) const {
    if (markerCountLimit <= 0) {
        return true;
    }
    return markerSelection.find(c.getSimpleIdLayer()) != markerSelection.end();
}

void Calib::purgeRecursionDeeperThan(int level) {
    for (auto& it : data) {
        it.second.purgeRecursionDeeperThan(level);
    }
}

void Calib::autoScaleCornerIds() {
    int max_main_marker_scale = 1;
    for (auto const& it : data) {
        max_main_marker_scale = std::max(max_main_marker_scale, it.second.getCornerIdFactorFromMainMarkers());
    }
    for (auto & it : data) {
        int const main_marker_scale = it.second.getCornerIdFactorFromMainMarkers();
        if (main_marker_scale < 1) {
            throw std::runtime_error("Invalid main marker scale: " + std::to_string(main_marker_scale));
        }
        if (main_marker_scale < max_main_marker_scale) {
            if ((max_main_marker_scale % main_marker_scale) != 0) {
                throw std::runtime_error ("max main marker scale not divisible by main marker scale");
            }
            int const factor = max_main_marker_scale / main_marker_scale;
            clog::L("Calib::autoScaleCornerIds", 2) << "Scaling ids of " << it.first << " by " << factor;
            it.second.scaleIDs(factor);
        }
    }
    cornerIdFactor = max_main_marker_scale;
}

void Calib::purgeUnlikelyByImageSize() {
    size_t removed = 0;
    runningstats::RunningStats removed_stats;
    for (auto& it : data) {
        size_t const local_removed = it.second.purgeUnlikelyByImageSize(imageSize);
        removed += local_removed;
        removed_stats.push_unsafe(local_removed);
    }
    clog::L("Calib::purgeUnlikelyByImageSize", 2) << "Removed: " << removed << ", stats: " << removed_stats.print();
}

size_t Calib::countMainMarkers(const std::vector<Corner> &vec) {
    size_t result = 0;
    for (Corner const& c : vec) {
        if (c.layer == 0) {
            result++;
        }
    }
    return result;
}

void Calib::prepareCalibrationByName(const string &name) {
    if ("OpenCV" == name || "SimpleOpenCV" == name) {
        prepareOpenCVCalibration();
        return;
    }
    if ("OpenCVRO" == name || "SimpleOpenCVRO" == name) {
        prepareOpenCVROCalibration();
        return;
    }
    prepareCalibration();
}

void Calib::combineImages(const string &out_file) {
    cv::Mat_<uint16_t> result = cv::imread(data.begin()->first, cv::IMREAD_UNCHANGED);
    result.setTo(0);
    clog::L("Calib::combineImages", 2) << out_file;
    for (auto const& it : data) {
        std::cout << it.first << "  ";
    }
    std::cout << std::endl;
    std::cout << std::string(data.size(), '.') << std::endl;
    for (auto const& it : data) {
        if (it.second.size() < 3) {
            std::cout << "-" << std::flush;
            continue;
        }
        std::cout << "+" << std::flush;
        std::vector<Corner> const& corners = it.second.getCorners();
        std::vector<cv::Point> points, hull;
        for (Corner const& c : corners) {
            points.push_back(c.p);
        }
        cv::convexHull(points, hull);
        cv::Mat_<uint8_t> mask(result.size(), uint8_t(0));
        cv::fillConvexPoly(mask, hull.data(), hull.size(), cv::Scalar(255));
        cv::Mat_<uint16_t> const img = cv::imread(it.first, cv::IMREAD_UNCHANGED);
        for (int ii = 0; ii < img.rows; ++ii) {
            for (int jj = 0; jj < img.cols; ++jj) {
                if (mask(ii,jj) > 127) {
                    result(ii, jj) = img(ii,jj);
                }
            }
        }
    }
    std::cout << std::endl;
    cv::imwrite(out_file, result);
}

std::vector<double> Calib::mat2vec(const cv::Mat_<double> &m) {
    std::vector<double> result;
    for (double const d : m) {
        result.push_back(d);
    }
    return result;
}

cv::Mat_<double> Calib::vec2squaremat(const std::vector<double> &vec) {
    int const n = std::round(std::sqrt(vec.size()));
    assert(vec.size() == size_t(n*n));

    cv::Mat_<double> m(n, n, 0.0);
    for (int ii = 0; ii < n; ++ii) {
        for (int jj = 0; jj < n; ++jj) {
            m(ii, jj) = vec[n*ii+jj];
        }
    }
    return m;
}

void Calib::scaleSquareMatVec(std::vector<double> &vec, int const N) {
    if (vec.empty()) {
        vec.resize(N*N, 0.0);
        return;
    }
    int const N_old = std::round(std::sqrt(vec.size()));
    assert(size_t(N_old * N_old) == vec.size());
    if (N == N_old) {
        return;
    }
    cv::Mat_<double> mat = vec2squaremat(vec);
    assert(mat.rows == N_old);
    assert(mat.cols == N_old);
    cv::resize(mat, mat, cv::Size(N, N), 0, 0, cv::INTER_CUBIC);
    vec = mat2vec(mat);
    assert(vec.size() == size_t(N*N));
}

Calib::Calib() {
    //clog::L(__func__, 2) << "Number of concurrent threads: " << threads << std::endl;
}

void Calib::save(const string &filename) {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    fs << "calibration" << *this;
    fs.release();}

void Calib::setMinSNR(const double val) {
    min_snr = val;
}

void Calib::setUseRaw(const bool val) {
    use_raw = val;
}

void Calib::setMaxIter(const size_t val) {
    max_iter = val;
}

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
        std::vector<cv::Point2d> reprojections;
        std::vector<Corner> markers;
        calib.getReprojections(ii, markers, reprojections);
        cv::Mat img = readImage(imageFiles[ii], demosaic, libraw, useOnlyGreen);
        if (1 == img.channels()) {
            cv::Mat _img[3] = {img.clone(), img.clone(), img.clone()};
            cv::merge(_img, 3, img);
        }
        for (size_t jj = 0; jj < markers.size(); ++jj) {
            cv::Point2d const & marker = markers[jj].p;
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
    std::vector<std::string> invalid;
    for (auto & it : data) {
        CornerStore & store = it.second;
        store.purgeOutOfBounds(limits.x, limits.y, limits.x + limits.width, limits.y + limits.height);
        std::cout << '.' << std::flush;
        if (store.countMainMarkers() < 4) {
            invalid.push_back(it.first);
        }
    }
    std::cout << std::endl;
    for (std::string const& remove : invalid) {
        removeInputImage(remove);
    }
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
    return cv::norm(a.p - b.p);
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

    std::vector<cv::Point2d> reprojections;
    std::vector<Corner> markers;
    for (size_t ii = 0; ii < imageFiles.size(); ++ii) {
        std::string const& filename = imageFiles[ii];
        std::ofstream out(filename + "-" + calib_name + "-target-points.txt", std::ofstream::out);
        CornerStore const& store = data[filename];
        calib.getReprojections(ii, markers, reprojections);
        if (markers.size() != reprojections.size()) {
            throw std::runtime_error("Markers size (" + std::to_string(markers.size()) + ") doesn't match reprojections (" + std::to_string(reprojections.size()) + ") in exportPointClouds");
        }
        /*
        if (markers.size() != store.size()) {
            throw std::runtime_error("Markers size (" + std::to_string(markers.size()) + ") doesn't match store (" + std::to_string(store.size()) + ") in exportPointClouds");
        }
        */
        for (size_t jj = 0; jj < store.size(); ++jj) {
            if (outlier_threshold <= 0 || distance<cv::Point2d>(markers[jj].p, reprojections[jj]) < outlier_threshold) {
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

void Calib::swapPointsForRO(std::vector<cv::Point2f>& img,
                            std::vector<cv::Point3f>& obj,
                            std::vector<size_t>& ref) {
    if (img.size() < 2) {
        throw std::runtime_error("Image points vector empty or too small in Calib::swapPointsForRO");
    }
    if (img.size() != obj.size() || img.size() != ref.size()) {
        throw std::runtime_error("Vector sizes don't match in Calib::swapPointsForRO");
    }
    randutils::mt19937_rng rng;
    size_t const size = img.size();
    size_t tl_index = rng.variate<size_t, std::uniform_int_distribution>(0, size-1);
    size_t tr_index = rng.variate<size_t, std::uniform_int_distribution>(0, size-1);
    size_t br_index = rng.variate<size_t, std::uniform_int_distribution>(0, size-1);
    cv::Point3f tl(obj[tl_index]), tr(obj[tr_index]), br(obj[br_index]);
    for (size_t ii = 0; ii < size; ++ii) {
        cv::Point3f &o = obj[ii];
        if (o.x <= tl.x && o.y >= tl.y) {
            tl = o;
            tl_index = ii;
        }
        if (o.x >= tr.x && o.y >= tr.y) {
            tr = o;
            tr_index = ii;
        }
        if (o.x >= br.x && o.y <= br.y) {
            br = o;
            br_index = ii;
        }
    }
    if (tl_index == tr_index) {
        throw std::runtime_error("Top left index matches top right index in Calib::swapPointsForRO");
    }
    if (tl_index == br_index) {
        throw std::runtime_error("Top left index matches bottom right index in Calib::swapPointsForRO");
    }
    if (tr_index == br_index) {
        throw std::runtime_error("Top right index matches bottom right index in Calib::swapPointsForRO");
    }
    cv::Point2f tl_img = img[tl_index], tr_img = img[tr_index], br_img = img[br_index];
    std::swap(img[0], img[tl_index]);
    std::swap(obj[0], obj[tl_index]);
    std::swap(ref[0], ref[tl_index]);

    std::swap(img[1], img[tr_index]);
    std::swap(obj[1], obj[tr_index]);
    std::swap(ref[1], ref[tr_index]);

    std::swap(img.back(), img[br_index]);
    std::swap(obj.back(), obj[br_index]);
    std::swap(ref.back(), ref[br_index]);

    img[0] = tl_img;
    obj[0] = tl;
    ref[0] = tl_index;

    img[1] = tr_img;
    obj[1] = tr;
    ref[1] = tr_index;

    img.back() = br_img;
    obj.back() = br;
    ref.back() = br_index;

    clog::L(__PRETTY_FUNCTION__, 2) << "Distances: "
                                    << "tl-tr: " << cv::norm(tl - tr)
                                    << ", tl-br: " << cv::norm(tl - br)
                                    << ", tr-br: " << cv::norm(tr-br);
}

void Calib::prepareOpenCVCalibration() {
    if (preparedOpenCVCalib && imagePoints.size() == data.size() && objectPoints.size() == data.size()) {
        return;
    }
    preparedOpenCVCalib = true;
    preparedOpenCVROCalib = false;
    preparedCalib = false;

    imagePoints = std::vector<std::vector<cv::Point2f> >(data.size());
    objectPoints = std::vector<std::vector<cv::Point3f> >(data.size());
    reduced_marker_references = std::vector<std::vector<cv::Scalar_<int> > >(data.size());
    imageFiles.resize(data.size());

    size_t ii = 0;
    for (std::pair<const std::string, CornerStore> const& it : data) {
        it.second.getMajorPoints(imagePoints[ii], objectPoints[ii], reduced_marker_references[ii], *this);
        imageFiles[ii] = it.first;
        ++ii;
    }
}

void Calib::prepareOpenCVROCalibration() {
    if (preparedOpenCVROCalib && imagePoints.size() == data.size() && objectPoints.size() == data.size()) {
        return;
    }
    preparedOpenCVROCalib = true;
    preparedOpenCVCalib = false;
    preparedCalib = false;

    CornerStore intersection = data.begin()->second.getMainMarkerAdjacent();
    for (std::pair<const std::string, CornerStore> const& it : data) {
        intersection.intersect(it.second);
    }

    corners = std::vector<std::vector<Corner> >(data.size());
    imagePoints = std::vector<std::vector<cv::Point2f> >(data.size());
    objectPoints = std::vector<std::vector<cv::Point3f> >(data.size());
    reduced_marker_references = std::vector<std::vector<cv::Scalar_<int> > >(data.size());
    imageFiles.resize(data.size());

    size_t ii = 0;
    for (std::pair<const std::string, CornerStore> const& it : data) {
        CornerStore copy = it.second;
        copy.intersect(intersection);
        copy.sort();
        copy.getPoints(imagePoints[ii], objectPoints[ii], reduced_marker_references[ii], corners[ii], *this);
        imageFiles[ii] = it.first;
        ++ii;
    }

    // Some sanity checks
    for (size_t ii = 1; ii < reduced_marker_references.size(); ++ii) {
        assert(reduced_marker_references[ii].size() == reduced_marker_references[0].size());
        for (size_t jj = 0; jj < reduced_marker_references[0].size(); ++jj) {
            assert(reduced_marker_references[ii][jj] == reduced_marker_references[0][jj]);
        }
    }

    third_fixed_point_index = 1;
    double best_val = 0;
    for (size_t ii = 1; ii+1 < objectPoints[0].size(); ++ii) {
        cv::Point3f const diff1 = objectPoints[0].front() - objectPoints[0][ii];
        double const length1 = std::sqrt(diff1.dot(diff1));

        cv::Point3f const diff2 = objectPoints[0].back() - objectPoints[0][ii];
        double const length2 = std::sqrt(diff2.dot(diff2));

        double const val = std::min(length1, length2);

        if (val > best_val) {
            best_val = val;
            third_fixed_point_index = ii;
        }
    }
    assert(third_fixed_point_index > 0);
    assert(third_fixed_point_index+1 < objectPoints[0].size());

    clog::L(__FUNCTION__, 2)
            << "Fixed points: " << objectPoints[0].front() << ", " << objectPoints[0][third_fixed_point_index] << ", " << objectPoints[0].back();

    clog::L(__FUNCTION__, 2)
            << "Distances between fixed points: "
            << "0-1: " << cv::norm(objectPoints[0].front() - objectPoints[0][third_fixed_point_index])
            << ", 1-2: " << cv::norm(objectPoints[0].back() - objectPoints[0][third_fixed_point_index])
            << ", 0-2: " << cv::norm(objectPoints[0].back() - objectPoints[0].front());

    clog::L(__FUNCTION__, 2) << "Number of markers used: " << objectPoints[0].size();
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

    res.x *= float(markerSize / cornerIdFactor) * (1.0 + x_factor/100.0); // Hacky adjustment for aspect ratio
    res.y *= float(markerSize / cornerIdFactor);

    return res;
}

Point3f Calib::getInitial3DCoord(const cv::Scalar &c, const double z) const {
    return getInitial3DCoord(cv::Point3f{float(c[0]), float(c[1]), float(c[2])}, z);
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
        if (!resolutionKnown && width > 0 && height > 0) {
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

    cv::Mat img_raw = convert16_8(getImageRaw(input_file));
    cv::imwrite("debug.png", img_raw);

    Marker::init();
    if (use_robust_main_marker_detection) {
        detect_robust(img_raw, result, use_rgb, 0, 10, effort);
    }
    else {
        detect(img_raw, result, use_rgb, 0, 10, effort);
    }
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
        clog::L("Calib::getImageScaled", 1) << "Input file empty, aborting." << std::endl;
        return {};
    }
    setImageSize(img);

    return scaleImage(img);
}

cv::Mat Calib::getImageRaw(std::string const& input_file) {
    cv::Mat img = readImage(input_file, demosaic, libraw, useOnlyGreen);
    clog::L(__PRETTY_FUNCTION__, 2) << "Read file " << input_file << ", got type " << type2str(img.type());
    if (img.empty()) {
        clog::L("Calib::getImageRaw", 1) << "Input file empty, aborting." << std::endl;
        return {};
    }
    setImageSize(img);

    return img;
}

vector<Corner> Calib::filterSNR(vector<Corner> const& in, double const threshold) {
    if (threshold <= 0) {
        return in;
    }
    std::vector<Corner> out;
    out.reserve(in.size());
    for (auto const& it : in) {
        if (it.layer == 0 || std::abs(it.getSigma() * it.snr) > threshold) {
            out.push_back(it);
        }
    }
    return out;
}

namespace {
void save_corners(std::string const &input_file, const vector<Corner> &corners_sub, std::vector<cv::Point3f> &tpoints) {
    std::string const output_file = input_file + ".points.yaml";
    if (fs::is_regular_file(output_file)) {
        return;
    }
    FileStorage store(output_file, FileStorage::WRITE);

    //    std::vector<cv::Point3f> tpoints;
    std::vector<cv::Point3f> ipoints;

    for(size_t i=0; i<corners_sub.size(); i++) {
        cv::Point3f ip = cv::Point3f(corners_sub[i].p.x, corners_sub[i].p.y, corners_sub[i].layer);
        ipoints.push_back(ip);
    }

    store << "target_points" << tpoints;
    store << "image_points" << ipoints;
}
}

void Calib::checkHDMPointcache(std::string const& input_file, std::vector<Corner> const& corners) {
    std::vector<cv::Point3f> tpoints;
    for(auto &c : corners)
        tpoints.push_back(cv::Point3f(c.id.x, c.id.y, c.page));
    save_corners(input_file, corners, tpoints);
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
        result = filterSNR(result, min_snr);
        checkHDMPointcache(input_file, result);
        return result;
    }
    if (nullptr != is_clean) {
        *is_clean = false;
    }

    std::string const hdm_gz_cache_file = input_file + "-submarkers.hdmarker.gz";
    if (fs::is_regular_file(hdm_gz_cache_file)) {
        Corner::readFile(hdm_gz_cache_file, result);
        result = filterSNR(result, min_snr);
        checkHDMPointcache(input_file, result);
        return result;
    }

    std::string const cv_gz_cache_file = input_file + "-submarkers.yaml.gz";
    if (fs::is_regular_file(cv_gz_cache_file)) {
        Corner::readFile(cv_gz_cache_file, result);
        int width = 0, height = 0;
        result = readCorners(cv_gz_cache_file, width, height);
        if (!resolutionKnown && width > 0 && height > 0) {
            imageSize.width = width;
            imageSize.height = height;
            resolutionKnown = true;
        }
        Corner::writeGzipFile(hdm_gz_cache_file, result);
        result = filterSNR(result, min_snr);
        checkHDMPointcache(input_file, result);
        return result;
    }

    std::vector<Corner> main_markers = getMainMarkers(input_file, effort, demosaic, raw);

    cv::Mat img = getImageRaw(input_file);

    double msize = 1.0;
    refineRecursiveByPage(img, main_markers, result, recursionDepth, msize, use_three);
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

    result = filterSNR(result, min_snr);
    checkHDMPointcache(input_file, result);
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
        return img_scaled;
    }
    else if (img_scaled.depth() == CV_8U) {
        img_scaled *= std::floor(255.0 / max);
        return img_scaled;
    }
    if (img_scaled.channels() == 1 && (img_scaled.depth() == CV_16U || img_scaled.depth() == CV_16S)) {
        clog::L(__func__, 0) << "Input image is 1 channel, 16 bit, converting for painting to 8 bit." << std::endl;
        img_scaled.convertTo(img_scaled, CV_8UC1, 1.0 / 256.0);
    }
    return img_scaled;
}

cv::Mat Calib::convert16_8(cv::Mat const& img) {
    cv::Mat result = img.clone();
    if (result.depth() == 2) {
        double min = 0;
        double max = 0;
        cv::minMaxIdx(img, &min, &max);
        clog::L("Calib::convert16_8", 2) << "Min/max value " << min << " / " << max;
        int const under_exposure_factor = std::floor(64*256 / max);
        int factor = 1;
        while (2*factor <= under_exposure_factor) {
            factor *= 2;
        }
        factor = under_exposure_factor;
        if (factor > 1) {
            clog::L("Calib::convert16_8", 2) << "Up-scaling by factor " << factor;
            result *= factor;
        }
        int downscale_factor = 1;
        cv::minMaxIdx(img, &min, &max);
        while (max / downscale_factor > 100*256) {
            downscale_factor++;
        }
        if (downscale_factor > 1) {
            //clog::L("Calib::convert16_8", 2) << "Down-scaling by factor " << downscale_factor;
            //result /= downscale_factor;
        }
        result.convertTo(result, CV_8UC1, 1.0 / 256.0);
    }
    return result;
}

void Calib::printCornerStats(std::vector<Corner> const& vec) {
    std::map<std::string, size_t> stats;
    std::map<int, size_t> stats_page;
    for (Corner const& c : vec) {
        std::string key = std::to_string(c.layer) + "/" + std::to_string(c.color);
        stats[key]++;
        stats_page[c.page]++;
    }
    std::cout << "Stats for layer/color combinations:" << std::endl;
    for (auto const& it : stats) {
        std::cout << it.first << ": " << it.second << std::endl;
    }
    std::cout << "Count by page:" << std::endl;
    for (auto const& it : stats_page) {
        std::cout << it.first << ": " << it.second << std::endl;
    }
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
            setImageSize(read_raw_imagesize(input_file));
        }
        else {
            img = getImageRaw(input_file);
            setImageSize(img);
        }
        resolutionKnown = true;
    }
    cv::Mat img_scaled;

    if (recursionDepth > 0) {
        printCornerStats(submarkers);
    }
    else {
        printCornerStats(corners);
    }

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
            paintSubmarkersRMS_SNR(input_file, submarkers, main_markers_area.size(), 1);
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
            paintSubmarkersRMS_SNR(input_file, submarkers, paint_submarkers.size(), paint_size_factor);
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
        img = cv::imread(input_file, cv::IMREAD_UNCHANGED);
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

double Calib::openCVCalib(bool const simple, bool const RO) {
    if (RO) {
        prepareOpenCVROCalibration();
    }
    else {
        prepareOpenCVCalibration();
    }

    if (imageSize.width <= 0 || imageSize.height <= 0) {
        setImageSize(readImage(imageFiles[0], use_raw, use_raw, useOnlyGreen).size());
    }

    std::string const name = std::string(simple ? "Simple" : "") + "OpenCV" + (RO ? "RO" : "");
    bool has_precalib = false;
    if (simple) {
        if (RO) {
            has_precalib = hasCalibName("SimpleOpenCVRO");
        }
        else {
            has_precalib = hasCalibName("SimpleOpenCV");
        }
    }
    else {
        if (RO) {
            if (!hasCalibName("OpenCVRO") && hasCalibName("SimpleOpenCVRO")) {
                calibrations["OpenCVRO"] = calibrations["SimpleOpenCV"];
            }
            else if (!hasCalibName("OpenCVRO") && hasCalibName("SimpleOpenCV")) {
                calibrations["OpenCVRO"] = calibrations["SimpleOpenCV"];
            }
        }
        else {
            if (!hasCalibName("OpenCV") && hasCalibName("SimpleOpenCV")) {
                calibrations["OpenCV"] = calibrations["SimpleOpenCV"];
            }
        }
    }
    CalibResult & res = calibrations[name];
    res.calib = this;
    res.name = name;

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

    runningstats::QuantileStats<float> correction_stats;

    double result_err = -1;
    if (RO) {
        //swapPointsForRO(imagePoints[0], objectPoints[0], reduced_marker_references[0]);
        std::vector<cv::Point3f> new_object_points;
        assert(data.size() == imagePoints.size());
        assert(objectPoints.size() == imagePoints.size());
        assert(objectPoints.size() == reduced_marker_references.size());
        // Transfer the objectPointCorrections to the new_object_points vector.
        for (size_t ii = 0; ii < reduced_marker_references[0].size(); ++ii) {
            auto it = res.objectPointCorrections.find(reduced_marker_references[0][ii]);
            cv::Point3f pt = objectPoints[0][ii];
            if (it != res.objectPointCorrections.end()) {
                pt += it->second;
            }
            new_object_points.push_back(pt);
        }
        {
            runningstats::QuantileStats<float> reprojection_errors;
            for (size_t ii = 0; ii < objectPoints.size(); ++ii) {
                std::vector<cv::Point2f> img_pts_out;
                cv::projectPoints (objectPoints[ii],
                                   res.rvecs[ii],
                                   res.tvecs[ii],
                                   res.cameraMatrix,
                                   res.distCoeffs,
                                   img_pts_out
                                   );
                assert(img_pts_out.size() == imagePoints[ii].size());
                for (size_t jj = 0; jj < img_pts_out.size(); ++jj) {
                    reprojection_errors.push_unsafe(cv::norm(img_pts_out[jj] - imagePoints[ii][jj]));
                }
            }
            clog::L(__FUNCTION__, 2) << "Errors pre-optimization: " << reprojection_errors.print() << ", median: " << reprojection_errors.getMedian();
        }
        clog::L(__FUNCTION__, 2) << "Third fixed point index: " << third_fixed_point_index;
        result_err = cv::calibrateCameraRO (
                    objectPoints,
                    imagePoints,
                    imageSize,
                    third_fixed_point_index,
                    res.cameraMatrix,
                    res.distCoeffs,
                    res.rvecs,
                    res.tvecs,
                    new_object_points,
                    res.stdDevIntrinsics,
                    res.stdDevExtrinsics,
                    res.stdDevObjectPoints,
                    res.perViewErrors,
                    flags
                    );
        assert(objectPoints[0].size() == new_object_points.size());
        assert(objectPoints[0].size() == reduced_marker_references[0].size());
        for (size_t ii = 0; ii < objectPoints.size(); ++ii) {
            assert(objectPoints[ii].size() == reduced_marker_references[0].size());
            assert(objectPoints[0].size() == reduced_marker_references[0].size());
        }
        {
            runningstats::QuantileStats<float> reprojection_errors;
            for (size_t ii = 0; ii < objectPoints.size(); ++ii) {
                std::vector<cv::Point2f> img_pts_out;
                cv::projectPoints (new_object_points,
                                   res.rvecs[ii],
                                   res.tvecs[ii],
                                   res.cameraMatrix,
                                   res.distCoeffs,
                                   img_pts_out
                                   );
                assert(img_pts_out.size() == imagePoints[ii].size());
                for (size_t jj = 0; jj < img_pts_out.size(); ++jj) {
                    reprojection_errors.push_unsafe(cv::norm(img_pts_out[jj] - imagePoints[ii][jj]));
                }
            }
            clog::L(__FUNCTION__, 2) << "Errors post-optimization: " << reprojection_errors.print() << ", median: " << reprojection_errors.getMedian();
        }
        // Transfer the new object point locations to objectPointCorrections map.
        for (size_t ii = 0; ii < new_object_points.size(); ++ii) {
            hdmarker::Corner c;
            bool const has_marker = data.begin()->second.hasIDLevel(reduced_marker_references[0][ii], c);
            assert(has_marker);
            cv::Point3f init_pt = getInitial3DCoord(c);
            cv::Point3f init_compare_pt = objectPoints[0][ii];
            assert(std::abs(init_pt.x - init_compare_pt.x) < 1e-4);
            assert(std::abs(init_pt.y - init_compare_pt.y) < 1e-4);
            assert(std::abs(init_pt.z - init_compare_pt.z) < 1e-4);
            cv::Point3f correction = new_object_points[ii] - init_compare_pt;
            res.objectPointCorrections[getSimpleIdLayer(c)] = correction;
            correction_stats.push_unsafe(cv::norm(correction));
        }
    }
    else {
        result_err = cv::calibrateCamera (
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
    }

    clog::L(__func__, 1) << "Median correction: " << correction_stats.print();
    clog::L(__func__, 1) << "RMSE: " << result_err << std::endl
                         << "Camera Matrix: " << std::endl << res.cameraMatrix << std::endl;
    clog::L(__func__, 1) << "per-view-errors: " << res.perViewErrors;
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
        return openCVCalib(false, false);
    }
    if ("OpenCVRO" == name) {
        return openCVCalib(false, true);
    }
    if ("SimpleOpenCV" == name) {
        return openCVCalib(true, false);
    }
    if ("SimpleOpenCVRO" == name) {
        return openCVCalib(true, true);
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
    if ("SemiFlexible" == name) {
        return CeresCalibSemiFlexibleTarget(outlier_threshold);
    }
    if ("FlexibleN" == name.substr(0, std::string("FlexibleN").size())) {
        int const num = std::stoi(name.substr(std::string("FlexibleN").size()));
        switch (num) {
        case  2: return CeresCalibFlexibleTargetN< 2>(outlier_threshold);
        case  4: return CeresCalibFlexibleTargetN< 4>(outlier_threshold);
        case  6: return CeresCalibFlexibleTargetN< 6>(outlier_threshold);
        case  8: return CeresCalibFlexibleTargetN< 8>(outlier_threshold);
        case 10: return CeresCalibFlexibleTargetN<10>(outlier_threshold);
        case 12: return CeresCalibFlexibleTargetN<12>(outlier_threshold);
        case 14: return CeresCalibFlexibleTargetN<14>(outlier_threshold);
        case 16: return CeresCalibFlexibleTargetN<16>(outlier_threshold);
        }
    }
    if ("FlexibleOdd" == name.substr(0, std::string("FlexibleOdd").size())) {
        int const num = std::stoi(name.substr(std::string("FlexibleOdd").size()));
        switch (num) {
        case  2: return CeresCalibFlexibleTargetOdd< 2>(outlier_threshold);
        case  4: return CeresCalibFlexibleTargetOdd< 4>(outlier_threshold);
        case  6: return CeresCalibFlexibleTargetOdd< 6>(outlier_threshold);
        case  8: return CeresCalibFlexibleTargetOdd< 8>(outlier_threshold);
        case 10: return CeresCalibFlexibleTargetOdd<10>(outlier_threshold);
        case 12: return CeresCalibFlexibleTargetOdd<12>(outlier_threshold);
        case 14: return CeresCalibFlexibleTargetOdd<14>(outlier_threshold);
        case 16: return CeresCalibFlexibleTargetOdd<16>(outlier_threshold);
        }
    }
    if ("Spline-" == name.substr(0, std::string("Spline-").size())) {
        int const num = std::stoi(name.substr(std::string("Spline-").size()));
        switch (num) {
        case  3: return CeresCalibFlexibleTargetSpline< 3, 3>(outlier_threshold);
        case  5: return CeresCalibFlexibleTargetSpline< 5, 3>(outlier_threshold);
        case  7: return CeresCalibFlexibleTargetSpline< 7, 3>(outlier_threshold);
        case  9: return CeresCalibFlexibleTargetSpline< 9, 3>(outlier_threshold);
        case 11: return CeresCalibFlexibleTargetSpline<11, 3>(outlier_threshold);
        case 13: return CeresCalibFlexibleTargetSpline<13, 3>(outlier_threshold);
        case 15: return CeresCalibFlexibleTargetSpline<15, 3>(outlier_threshold);
        case 17: return CeresCalibFlexibleTargetSpline<17, 3>(outlier_threshold);
        case 19: return CeresCalibFlexibleTargetSpline<19, 3>(outlier_threshold);
        case 21: return CeresCalibFlexibleTargetSpline<21, 3>(outlier_threshold);
        }
    }
    throw std::runtime_error(std::string("Calib type ") + name + " unknown");
}

std::string Calib::printVec(std::vector<double> const& vec) {
    std::stringstream result;
    if (vec.empty()) {
        return "";
    }
    result << vec[0];
    for (size_t ii = 1; ii < vec.size(); ++ii) {
        result << ", " << vec[ii];
    }
    return result.str();
}

void Calib::setPlotMarkers(bool plot) {
    plotMarkers = plot;
}

void Calib::setPlotSubMarkers(bool plot) {
    plotSubMarkers = plot;
}

void Calib::setImageSize(const Mat &img) {
    setImageSize(img.size());
}

void Calib::setImageSize(const Size &s) {
    if (s.width > 0 && s.height > 0) {
        imageSize = s;
        resolutionKnown = true;
    }
}

Size Calib::getImageSize() const {
    return imageSize;
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

cv::Scalar_<int> Calib::getSimpleIdLayer(const Corner &marker) {
    return {marker.id.x, marker.id.y, marker.page, marker.layer};
}

uint64_t Calib::getIdHash(const Corner &marker) {
    return uint64_t(marker.layer)
            + (uint64_t(marker.id.x) << 4)
            + (uint64_t(marker.id.y) << (4+16))
            + (uint64_t(marker.page) << (4+16+16));
    //            ^^^^ <4bit        ^^^ < 16 bit          ^^^ < 16 bit           ^^^^ < 10 bit
}

void Calib::findOutliers(
        std::string const& calib_name,
        const double threshold,
        const size_t image_index,
        std::vector<hdmarker::Corner> & outliers) {
    std::vector<cv::Point2d> projections;
    std::vector<Corner> markers;
    calibrations[calib_name].getReprojections(image_index, markers, projections);

    runningstats::RunningStats inlier_stats, outlier_stats;
    for (size_t ii = 0; ii < markers.size(); ++ii) {
        double const error = distance<cv::Point2d>(markers[ii].p, projections[ii]);
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
    _src += calib.objectPointCorrections[getSimpleIdLayer(c)];
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

template<class T>
bool cmpScalar::operator()(const cv::Scalar_<T> &a, const cv::Scalar_<T> b) const {
    for (size_t ii = 0; ii < 3; ++ii) {
        if (a[ii] != b[ii]) return a[ii] < b[ii];
    }
    return a[3] < b[3];
}

template bool cmpScalar::operator()(const cv::Scalar_<int> &a, const cv::Scalar_<int> b) const;

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

string type2str(int type) {
std::string r;

uchar depth = type & CV_MAT_DEPTH_MASK;
uchar chans = 1 + (type >> CV_CN_SHIFT);

switch ( depth ) {
case CV_8U:  r = "8U"; break;
case CV_8S:  r = "8S"; break;
case CV_16U: r = "16U"; break;
case CV_16S: r = "16S"; break;
case CV_32S: r = "32S"; break;
case CV_32F: r = "32F"; break;
case CV_64F: r = "64F"; break;
default:     r = "User"; break;
}

r += "C";
r += (chans+'0');

return r;
}

bool CalibResult::isExpandedType() {
for (std::string const& prefix : {"FlexibleN", "FlexibleOdd", "SemiFlexibleN", "SemiFlexibleOdd"}) {
if (Calib::startsWith(name, prefix)) {
return true;
}
}
return false;
}

FixedValues::FixedValues(const bool all):
focal(all),
principal(all),
dist(all),
objCorr(all),
rvecs(all),
tvecs(all),
x_factor(all){}

} // namespace hdcalib
