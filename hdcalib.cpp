#include "hdcalib.h"

#include "gnuplot-iostream.h"

#include <ceres/ceres.h>
#include <opencv2/optflow.hpp>
#include <opencv2/video/tracking.hpp>
#include <runningstats/runningstats.h>

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

void Calib::removeOutliers(const double threshold) {
    std::vector<hdmarker::Corner> outliers;
    for (size_t ii = 0; ii < data.size(); ++ii) {
        findOutliers(
                    threshold,
                    ii,
                    outliers);
    }
    if (outliers.empty()) {
        return;
    }
    CornerStore subtrahend(outliers);
    std::stringstream msg;
    msg << "Oulier percentage by image:" << std::endl;
    runningstats::RunningStats percent_stats;
    for (auto& it : data) {
        size_t const before = it.second.size();
        it.second.difference(subtrahend);
        size_t const after = it.second.size();
        double const percent = (double(before - after)/before)*100.0;
        percent_stats.push(percent);
        msg << it.first << ": removed " << (before-after) << " out of " << before << " corners (" << percent << "%)" << std::endl;
    }
    msg << "Removeal percentage stats: " << percent_stats.print() << std::endl;
    if (verbose) {
        std::cout << msg.str() << std::endl;
    }
    prepareCalibration();
}

void Calib::printObjectPointCorrectionsStats(
        const std::map<Point3i, Point3f, cmpSimpleIndex3<Point3i> > &corrections) const {
    runningstats::RunningStats dx, dy, dz, abs_dx, abs_dy, abs_dz, length;
    for (std::pair<cv::Point3i, cv::Point3f> const& it : corrections) {
        dx.push(it.second.x);
        dy.push(it.second.y);
        dz.push(it.second.z);

        abs_dx.push(std::abs(it.second.x));
        abs_dy.push(std::abs(it.second.y));
        abs_dz.push(std::abs(it.second.z));

        length.push(std::sqrt(it.second.dot(it.second)));
    }
    std::cout << "Object point correction stats: " << std::endl
              << "dx: " << dx.print() << std::endl
              << "dy: " << dy.print() << std::endl
              << "dz: " << dz.print() << std::endl
              << "abs(dx): " << abs_dx.print() << std::endl
              << "abs(dy): " << abs_dy.print() << std::endl
              << "abs(dz): " << abs_dz.print() << std::endl
              << "length: " << length.print() << std::endl
              << std::endl;
}

void Calib::getReprojections(
        const size_t ii,
        std::vector<Point2d> &markers,
        std::vector<Point2d> &reprojections) {

    auto const& imgPoints = imagePoints[ii];
    auto const& objPoints = objectPoints[ii];
    std::string const& filename = imageFiles[ii];
    CornerStore const& store = data[filename];
    cv::Mat const& rvec = rvecs[ii];
    cv::Mat const& tvec = tvecs[ii];

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
    for (size_t jj = 0; jj < 14; ++jj) {
        dist[jj] = distCoeffs.cols > jj ? distCoeffs(jj) : 0;
    }

    double focal[2] = {cameraMatrix(0,0), cameraMatrix(1,1)};
    double principal[2] = {cameraMatrix(0,2), cameraMatrix(1,2)};

    std::vector<std::vector<double> > data;
    for (size_t jj = 0; jj < imgPoints.size(); ++jj) {
        cv::Point3f current_objPoint = objPoints[jj] + objectPointCorrections[getSimpleId(store.get(jj))];

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

void Calib::prepareCalibration()
{
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

Calib::Calib() {

}

void Calib::setRecursionDepth(int _recursionDepth) {
    recursionDepth = _recursionDepth;
}

void Calib::scaleCornerIds(std::vector<Corner> &corners, int factor) {
    for (hdmarker::Corner& c: corners) {
        c.id *= factor;
    }
}

void Calib::piecewiseRefinement(cv::Mat & img, const std::vector<Corner> &in, std::vector<Corner> &out, int recursion_depth, double &markerSize) {
    CornerStore store(in);
    for (hdmarker::Corner const& it : in) {
        if (it.id.x > 30 || it.id.y > 30) {
            continue;
        }
        std::vector<hdmarker::Corner> local_corners;
        local_corners.push_back(it);
        {
            hdmarker::Corner copy(it);
            copy.id.x += 1;
            std::vector<hdmarker::Corner> res = store.findByID(copy);
            if (res.empty() || res.front().id != copy.id || res.front().page != copy.page) continue;
            local_corners.push_back(res.front());
        }
        {
            hdmarker::Corner copy(it);
            copy.id.y += 1;
            std::vector<hdmarker::Corner> res = store.findByID(copy);
            if (res.empty() || res.front().id != copy.id || res.front().page != copy.page) continue;
            local_corners.push_back(res.front());
        }
        {
            hdmarker::Corner copy(it);
            copy.id.x += 1;
            copy.id.y += 1;
            std::vector<hdmarker::Corner> res = store.findByID(copy);
            if (res.empty() || res.front().id != copy.id || res.front().page != copy.page) continue;
            local_corners.push_back(res.front());
        }
        cv::Rect limits(it.id.x, it.id.y, 1, 1);
        std::vector<hdmarker::Corner> local_submarkers;
        hdmarker::refine_recursive(img, in, local_submarkers, recursion_depth, &markerSize,
                                   nullptr, // cv::Mat paint *
                                   nullptr, // bool * mask_2x2
                                   it.page, // page
                                   limits
                                   );
        out.insert( out.end(), local_submarkers.begin(), local_submarkers.end() );
    }
    if (out.size() < in.size()) {
        out = in;
    }
}

void Calib::refineRecursiveByPage(Mat &img, const std::vector<Corner> &in, std::vector<Corner> &out, int recursion_depth, double &markerSize) {
    std::map<int, std::vector<hdmarker::Corner> > pages;
    for (const hdmarker::Corner& c : in) {
        pages[c.page].push_back(c);
    }
    out.clear();
    double _markerSize = markerSize;
    for (const auto& it : pages) {
        _markerSize = markerSize;
        std::vector<hdmarker::Corner> _out;
        hdmarker::refine_recursive(img, it.second, _out, recursion_depth, &_markerSize);
        out.insert(out.end(), _out.begin(), _out.end());
    }
    markerSize = _markerSize;
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
    result.reserve(in.cols*in.rows);
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
    return {in.x, in.y, in.z};
}

Point3f Calib::vec2point3f(const std::vector<double> &in) {
    if (in.size() < 3) {
        throw std::runtime_error("Less than 3 elements in input vector.");
    }
    return cv::Point3f(in[0], in[1], in[2]);
}

Point3f Calib::getInitial3DCoord(const Corner &c, const double z) const {
    return getInitial3DCoord(getSimpleId(c), z);
}

Point3f Calib::getInitial3DCoord(const Point3i &c, const double z) const {
    cv::Point3f res(c.x, c.y, z);
    switch (c.z) {
    case 1: res.x += cornerIdFactor * 32; break;
    case 2: res.x += cornerIdFactor * 64; break;

    case 3: res.y += cornerIdFactor * 32; break;
    case 4: res.y += cornerIdFactor * 32; res.x += cornerIdFactor * 32; break;
    case 5: res.y += cornerIdFactor * 32; res.x += cornerIdFactor * 64; break;

    case 7: res.x += cornerIdFactor * 32; break;
    }

    res.x *= markerSize / cornerIdFactor;
    res.y *= markerSize / cornerIdFactor;

    return res;
}

void Calib::keepCommonCorners_delete() {
    CornerStore _delete;
    CornerStore _union = getUnion();

    for (size_t ii = 0; ii < _union.size(); ++ii) {
        hdmarker::Corner const c = _union.get(ii);
        for (auto const& it : data) {
            if (!(it.second.hasID(c))) {
                _delete.push_conditional(c);
                break;
            }
        }
    }

    for (auto& it : data) {
        bool found_delete = false;
        CornerStore replacement;
        for (size_t ii = 0; ii < it.second.size(); ++ii) {
            if (_delete.hasID(it.second.get(ii))) {
                found_delete = true;
            }
            else {
                replacement.push_conditional(it.second.get(ii));
            }
        }
        if (found_delete) {
            it.second = replacement;
        }
    }
}

void Calib::keepCommonCorners_intersect() {
    if (data.empty() || data.size() < 2) {
        return;
    }
    std::string const last = std::prev(data.end())->first;
    if (data.size() == 2) {
        CornerStore::intersect(data.begin()->second, data[last]);
    }
    std::string prev = last;
    for (auto& it : data) {
        it.second.intersect(data[prev]);
        prev = it.first;
    }
    prev = std::prev(data.end())->first;
    for (auto& it : data) {
        it.second.intersect(data[prev]);
        prev = it.first;
    }
}

void Calib::keepCommonCorners() {
    keepCommonCorners_intersect();
}

void Calib::addInputImage(const string filename, const std::vector<Corner> &corners) {
    CornerStore & ref = data[filename];
    ref.replaceCorners(corners);
    ref.clean(cornerIdFactor);
}

template<class T, class T1, class T2>
void Calib::insertSorted(std::vector<T>& a, std::vector<T1>& b, std::vector<T2>& c) {
    if (a.size() != b.size() || a.size() != c.size()) {
        throw std::runtime_error(std::string("Sizes of arrays do not match: ")
                                 + std::to_string(a.size()) + ", "
                                 + std::to_string(b.size())
                                 + std::to_string(c.size()));
    }
    if (a.size() < 2) {
        return;
    }
    for (size_t ii = a.size()-1; ii > 0; --ii) {
        if (a[ii] < a[ii-1]) {
            std::swap(a[ii], a[ii-1]);
            std::swap(b[ii], b[ii-1]);
            std::swap(c[ii], c[ii-1]);
        }
        else {
            return;
        }
    }
}

template void Calib::insertSorted(std::vector<std::string> &, std::vector<std::string> &, std::vector<std::string> &);

void Calib::addInputImageAfterwards(const string filename, const std::vector<Corner> &corners) {
    prepareCalibration();

    rvecs.push_back(cv::Mat());
    tvecs.push_back(cv::Mat());
    imageFiles.push_back(filename);


    insertSorted(imageFiles, rvecs, tvecs);

    CornerStore & ref = data[filename];
    ref.replaceCorners(corners);
    ref.clean(cornerIdFactor);

    prepareCalibration();

    size_t index = 0;
    for (; index < imageFiles.size(); ++index) {
        if (filename == imageFiles[index]) {
            break;
        }
    }

    bool const success = cv::solvePnP (
                objectPoints[index],
                imagePoints[index],
                cameraMatrix,
                distCoeffs,
                rvecs[index],
                tvecs[index]);

}

void Calib::addInputImage(const string filename, const std::vector<Corner> &corners, cv::Mat const& rvec, cv::Mat const& tvec) {
    CornerStore & ref = data[filename];
    ref.replaceCorners(corners);
    ref.clean(cornerIdFactor);
    rvecs.push_back(rvec);
    tvecs.push_back(tvec);
}

void Calib::addInputImage(const string filename, const CornerStore &corners) {
    CornerStore & ref = data[filename];
    ref = corners;
    ref.clean(cornerIdFactor);
}

CornerStore Calib::get(const string filename) const {
    Store_T::const_iterator it = data.find(filename);
    if (it != data.end()) {
        return it->second;
    }
    return CornerStore();
}

Mat Calib::normalize_raw_per_channel(const Mat &input) {
    cv::Mat result = input.clone();

    normalize_raw_per_channel_inplace(result);
    return result;
}

void Calib::normalize_raw_per_channel_inplace(Mat &input) {
    unsigned short max_r = 0, max_g = 0, max_b = 0;
    for (int ii = 0; ii < input.rows; ++ii) {
        unsigned short const * const row = input.ptr<unsigned short>(ii);
        for (int jj = 0; jj < input.cols; ++jj) {
            switch (color(ii, jj)) {
            case 'R': max_r = std::max(max_r, row[jj]); break;
            case 'G': max_g = std::max(max_g, row[jj]); break;
            case 'B': max_b = std::max(max_b, row[jj]); break;
            default: throw std::runtime_error("Color value not R,G,B");
            }
        }
    }
    const double f_r = 65535.0/max_r;
    const double f_g = 65535.0/max_g;
    const double f_b = 65535.0/max_b;

    for (int ii = 0; ii < input.rows; ++ii) {
        unsigned short * row = input.ptr<unsigned short>(ii);
        for (int jj = 0; jj < input.cols; ++jj) {
            switch (color(ii, jj)) {
            case 'R': row[jj] = cv::saturate_cast<unsigned short>(std::round(row[jj] * f_r)); break;
            case 'G': row[jj] = cv::saturate_cast<unsigned short>(std::round(row[jj] * f_g)); break;
            case 'B': row[jj] = cv::saturate_cast<unsigned short>(std::round(row[jj] * f_b)); break;
            default: throw std::runtime_error("Color value not R,G,B");
            }
        }
    }

}

void printHist(std::ostream& out, runningstats::Histogram const& h, double const threshold = 0) {
    auto hist = h.getRelativeHist();
    double threshold_sum = 0;
    double last_thresh_key = hist.front().first;
    double prev_key = last_thresh_key;
    for (auto const& it : hist) {
        if (it.second > threshold) {
            if (threshold_sum > 0) {
                if (last_thresh_key == prev_key) {
                    out << prev_key << ": " << threshold_sum << std::endl;
                }
                else {
                    out << last_thresh_key << " - " << prev_key << ": " << threshold_sum << std::endl;
                }
            }
            out << it.first << ": " << it.second << std::endl;
            threshold_sum = 0;
            last_thresh_key = it.first;
        }
        else {
            threshold_sum += it.second;
            if (threshold_sum > threshold) {
                out << last_thresh_key << " - " << prev_key << ": " << threshold_sum << std::endl;
                threshold_sum = 0;
                last_thresh_key = it.first;
            }
        }
        prev_key = it.first;
    }
}

Mat Calib::read_raw(const string &filename) {
    LibRaw RawProcessor;


    auto& S = RawProcessor.imgdata.sizes;
    auto& OUT = RawProcessor.imgdata.params;

    int ret;
    if ((ret = RawProcessor.open_file(filename.c_str())) != LIBRAW_SUCCESS) {
        throw std::runtime_error(std::string("Cannot open file ") + filename + ", "
                                 + libraw_strerror(ret) + "\r\n");
    }
    if (verbose) {
        printf("Image size: %dx%d\nRaw size: %dx%d\n", S.width, S.height, S.raw_width, S.raw_height);
        printf("Margins: top=%d, left=%d\n", S.top_margin, S.left_margin);
    }

    if ((ret = RawProcessor.unpack()) != LIBRAW_SUCCESS) {
        throw std::runtime_error(std::string("Cannot unpack file ") + filename + ", "
                                 + libraw_strerror(ret) + "\r\n");
    }

    if (verbose) {
        printf("Unpacked....\n");
        std::cout << "Color matrix (top left corner):" << std::endl;
        for (size_t ii = 0; ii < 6 && ii < S.width; ++ii) {
            for (size_t jj = 0; jj < 6 && jj < S.height; ++jj) {
                std::cout << RawProcessor.COLOR(ii, jj) << ":"
                          << RawProcessor.imgdata.idata.cdesc[RawProcessor.COLOR(ii, jj)] << " ";
            }
            std::cout << std::endl;
        }
        runningstats::Histogram r(1),g(1),b(1);
        for (int jj = 0; jj < S.height; ++jj) {
            int global_counter = jj * S.raw_width;
            for (size_t ii = 0; ii < S.height; ++ii, ++global_counter) {
                int const value = RawProcessor.imgdata.rawdata.raw_image[global_counter];
                switch(RawProcessor.imgdata.idata.cdesc[RawProcessor.COLOR(ii, jj)]) {
                case 'R': r.push(value); break;
                case 'G': g.push(value); break;
                case 'B': b.push(value); break;
                }
            }
        }
        double const hist_threshold = 0.001;
        std::cout << "R-channel histogram:" << std::endl;
        printHist(std::cout, r, hist_threshold);
        std::cout << "G-channel histogram:" << std::endl;
        printHist(std::cout, g, hist_threshold);
        std::cout << "B-channel histogram:" << std::endl;
        printHist(std::cout, b, hist_threshold);

        std::cout << "model2: " << RawProcessor.imgdata.color.model2 << std::endl;
        std::cout << "UniqueCameraModel: " << RawProcessor.imgdata.color.UniqueCameraModel << std::endl;
        std::cout << "LocalizedCameraModel: " << RawProcessor.imgdata.color.LocalizedCameraModel << std::endl;

        std::cout << "desc: " << RawProcessor.imgdata.other.desc << std::endl;
        std::cout << "artist: " << RawProcessor.imgdata.other.artist << std::endl;

        std::cout << "make: " << RawProcessor.imgdata.idata.make << std::endl;
        std::cout << "model: " << RawProcessor.imgdata.idata.model << std::endl;
    }

    if (!(RawProcessor.imgdata.idata.filters || RawProcessor.imgdata.idata.colors == 1)) {
        throw std::runtime_error(
                    std::string("Only Bayer-pattern RAW files supported, file ") + filename
                    + " seems to have a different pattern.\n");
    }

    cv::Mat result(S.height, S.width, CV_16UC1);

    for (int jj = 0, global_counter = 0; jj < S.height ; jj++) {
        unsigned short * row = result.ptr<unsigned short>(jj);
        global_counter = jj * S.raw_width;
        for (int ii = 0; ii < S.width; ++ii, ++global_counter) {
            row[ii] = RawProcessor.imgdata.rawdata.raw_image[global_counter];
        }
    }

    if ("Sony" == std::string(RawProcessor.imgdata.idata.make)
            && "ILCE-7RM2" == std::string(RawProcessor.imgdata.idata.model)) {
        if (verbose) {
            std::cout << "Known camera detected, scaling result" << std::endl;
        }
        //result *= 4;
        double min = 0, max = 0;
        if (verbose) {
            cv::minMaxIdx(result, &min, &max);
            std::cout << "original min/max: " << min << " / " << max << std::endl;
        }
        result.forEach<uint16_t>([&](uint16_t& element, const int position[]) -> void
        {
            element *= 4;
        }
        );
        if (verbose) {
            cv::minMaxIdx(result, &min, &max);
            std::cout << "scaled min/max: " << min << " / " << max << std::endl;
        }
    }


    return result;
}

cv::Size Calib::read_raw_imagesize(const string &filename) {
    LibRaw RawProcessor;

    auto& S = RawProcessor.imgdata.sizes;

    int ret;
    if ((ret = RawProcessor.open_file(filename.c_str())) != LIBRAW_SUCCESS)
    {
        throw std::runtime_error(std::string("Cannot open file ") + filename + ", "
                                 + libraw_strerror(ret) + "\r\n");
    }
    if (verbose) {
        printf("Image size: %dx%d\nRaw size: %dx%d\n", S.width, S.height, S.raw_width, S.raw_height);
        printf("Margins: top=%d, left=%d\n", S.top_margin, S.left_margin);
    }

    return cv::Size(S.width, S.height);
}

void Calib::printObjectPointCorrectionsStats() {
    printObjectPointCorrectionsStats(objectPointCorrections);
}

void Calib::write(FileStorage &fs) const {

    fs << "{"
       << "cameraMatrix" << cameraMatrix;
    fs << "distCoeffs" << distCoeffs
       << "imageSize" << imageSize
       << "resolutionKnown" << resolutionKnown
       << "apertureWidth" << apertureWidth
       << "apertureHeight" << apertureHeight
       << "useOnlyGreen" << useOnlyGreen
       << "recursionDepth" << recursionDepth;

    fs << "images" << "[";
    for (size_t ii = 0; ii < imageFiles.size(); ++ii) {
        fs << "{";
        fs << "name" << imageFiles[ii]
              << "rvec" << rvecs[ii]
                 << "tvec" << tvecs[ii]
                    << "corners" << "[";
        CornerStore const & store = data.find(imageFiles[ii])->second;
        for (size_t jj = 0; jj < store.size(); ++jj) {
            fs << store.get(jj);
        }
        fs << "]";
        fs << "}";
    }
    fs << "]";

    fs << "objectPointCorrections" << "[";
    for (const auto& it : objectPointCorrections) {
        fs << "{"
           << "id" << it.first
           << "val" << it.second
           << "}";
    }
    fs << "]";

    fs << "}";
}

void Calib::read(const FileNode &node) {
    node["cameraMatrix"] >> cameraMatrix;
    node["distCoeffs"] >> distCoeffs;
    node["imageSize"] >> imageSize;
    node["resolutionKnown"] >> resolutionKnown;
    node["apertureWidth"] >> apertureWidth;
    node["apertureHeight"] >> apertureHeight;
    node["useOnlyGreen"] >> useOnlyGreen;
    node["recursionDepth"] >> useOnlyGreen;

    FileNode n = node["images"]; // Read string sequence - Get node
    if (n.type() != FileNode::SEQ) {
        throw std::runtime_error("Error while reading cached calibration result: Images is not a sequence. Aborting.");
    }

    for (FileNodeIterator it = n.begin(); it != n.end(); ++it) {
        std::vector<hdmarker::Corner> corners;
        cv::Mat rvec, tvec;
        std::string name;
        (*it)["name"] >> name;
        (*it)["rvec"] >> rvec;
        (*it)["tvec"] >> tvec;
        cv::FileNode corners_node = (*it)["corners"];
        if (corners_node.type() != FileNode::SEQ) {
            throw std::runtime_error(std::string("Error while reading cached calibration result for image ") + name + ": Corners is not a sequence. Aborting.");
        }
        for (FileNodeIterator corner_it = corners_node.begin(); corner_it != corners_node.end(); ++corner_it) { // Go through the node
            hdmarker::Corner c;
            *corner_it >> c;
            corners.push_back(c);
        }
        addInputImage(name, corners, rvec, tvec);
    }

    n = node["objectPointCorrections"]; // Read string sequence - Get node
    if (n.type() != FileNode::SEQ) {
        throw std::runtime_error("Error while reading cached calibration result: objectPointCorrections is not a sequence. Aborting.");
    }
    for (FileNodeIterator it = n.begin(); it != n.end(); ++it) {
        cv::Point3i id;
        cv::Point3f val;
        (*it)["id"] >> id;
        (*it)["val"] >> val;
        objectPointCorrections[id] = val;
    }

    prepareCalibration();
}

struct GridCost {
    int const row;
    int const col;

    cv::Vec3d const p;
    cv::Vec3d const p0;

    GridCost(int const _row, int const _col, cv::Vec3d const & _p, cv::Vec3d const & _p0) : row(_row), col(_col), p(_p), p0(_p0) {}

    template<class T>
    bool operator () (
            T const * const row_vec,
            T const * const col_vec,
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

    cv::Vec3d rect_rot;
    getRectificationRotation(rows, cols, images, rect_rot);
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
            std::cout << "row: " << row << ", col: " << col << ", counter: " << counter << ", name: " << images[counter] << std::endl;
            if (0 == col && 0 == row) {
                continue;
            }
            CornerStore const& current = data[images[counter]];
            size_t const id = getId(images[counter]);
            std::vector<GridCost* > & target_costs = cost_functions[images[counter]];
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

                cv::Vec3d src1, dst1;
                src1 = get3DPoint(src, rvecs[middle_id], tvecs[middle_id]);
                dst1 = get3DPoint(dst, rvecs[id], tvecs[id]);
                GridCost * cost = new GridCost(row, col, src1, dst1);
                ceres::CostFunction * cost_function = new ceres::AutoDiffCostFunction<GridCost, 3, 3, 3>(
                            cost
                            );
                problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(.5), row_vec.val, col_vec.val);
                target_costs.push_back(cost);
            }
        }
    }
    ceres::Solver::Options options;
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
                g_res[ii].push_unsafe(residuals[ii]);
                res[ii].push_unsafe(residuals[ii]);
                g_err[ii].push_unsafe(std::abs(residuals[ii]));
                err[ii].push_unsafe(std::abs(residuals[ii]));
            }
        }

        std::cout << "Residual stats for " << it.first << ":" << std::endl;
        for (size_t ii = 0; ii < 3; ++ii) {
            std::cout << res[ii].print() << std::endl;
        }
        std::cout << "Error stats for " << it.first << ":" << std::endl;
        for (size_t ii = 0; ii < 3; ++ii) {
            std::cout << err[ii].print() << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << "Residual stats for all images:" << std::endl;
    for (size_t ii = 0; ii < 3; ++ii) {
        std::cout << g_res[ii].print() << std::endl;
    }
    std::cout << "Error stats for all images:" << std::endl;
    for (size_t ii = 0; ii < 3; ++ii) {
        std::cout << g_err[ii].print() << std::endl;
    }
    std::cout << std::endl;

    double const row_vec_length = std::sqrt(row_vec.dot(row_vec));
    double const col_vec_length = std::sqrt(col_vec.dot(col_vec));
    std::cout << "row_vec: " << row_vec << ", length: " << row_vec_length << std::endl
              << "col_vec: " << col_vec << ", length: " << col_vec_length << std::endl;

    double cos_alpha = row_vec.dot(col_vec) / (row_vec_length * col_vec_length);
    std::cout << "cos alpha: " << cos_alpha << std::endl;
    std::cout << "angle: " << std::acos(cos_alpha)/M_PI*180. << std::endl;
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
            read_cache_success = true;
            FileStorage pointcache(pointcache_file, FileStorage::READ);
            FileNode n = pointcache["corners"]; // Read string sequence - Get node
            if (n.type() != FileNode::SEQ) {
                read_cache_success = false;
                throw std::runtime_error("Corners is not a sequence! FAIL");
            }

            FileNodeIterator it = n.begin(), it_end = n.end(); // Go through the node
            for (; it != it_end; ++it) {
                hdmarker::Corner c;
                *it >> c;
                corners.push_back(c);
            }
            if (pointcache["imageWidth"].isInt() && pointcache["imageHeight"].isInt()) {
                int width = 0, height = 0;
                pointcache["imageWidth"] >> width;
                pointcache["imageHeight"] >> height;
                if (0 != width && 0 != height) {
                    imageSize.width = width;
                    imageSize.height = height;
                    resolutionKnown = true;
                }
            }
        }
    }
    catch (const Exception& e) {
        std::cout << "Reading pointcache file failed with exception: " << std::endl
                  << e.what() << std::endl;
        read_cache_success = false;
    }
    if (read_cache_success) {
        corners = filter_duplicate_markers(corners);
        std::cout << "Got " << corners.size() << " corners from pointcache file" << std::endl;
    }

    bool read_submarkers_success = false;
    try {
        if (fs::exists(submarkers_file)) {
            read_submarkers_success = true;
            FileStorage pointcache(submarkers_file, FileStorage::READ);
            FileNode n = pointcache["corners"]; // Read string sequence - Get node
            if (n.type() != FileNode::SEQ) {
                read_submarkers_success = false;
                throw std::runtime_error("Corners is not a sequence! FAIL");
            }

            FileNodeIterator it = n.begin(), it_end = n.end(); // Go through the node
            for (; it != it_end; ++it) {
                hdmarker::Corner c;
                *it >> c;
                submarkers.push_back(c);
            }

            if (pointcache["imageWidth"].isInt() && pointcache["imageHeight"].isInt()) {
                int width = 0, height = 0;
                pointcache["imageWidth"] >> width;
                pointcache["imageHeight"] >> height;
                if (0 != width && 0 != height) {
                    imageSize.width = width;
                    imageSize.height = height;
                    resolutionKnown = true;
                }
            }
        }
    }
    catch (const Exception& e) {
        std::cout << "Reading pointcache file failed with exception: " << std::endl
                  << e.what() << std::endl;
        read_submarkers_success = false;
    }
    if (read_submarkers_success) {
        submarkers = filter_duplicate_markers(submarkers);
        std::cout << "Got " << submarkers.size() << " submarkers from submarkers file" << std::endl;
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

    if (plotMarkers || !read_cache_success || (recursionDepth > 0 && !read_submarkers_success)) {
        if (demosaic) {
            if (raw) {
                img = read_raw(input_file);
            }
            else {
                img = cv::imread(input_file, cv::IMREAD_GRAYSCALE);
            }
            setImageSize(img);
            double min_val = 0, max_val = 0;
            cv::minMaxIdx(img, &min_val, &max_val);
            std::cout << "Image min/max: " << min_val << " / " << max_val << std::endl;
            img = img * (65535 / max_val);

            cvtColor(img, img, COLOR_BayerBG2BGR); // RG BG GB GR
        }
        else {
            img = cv::imread(input_file);
            setImageSize(img);
        }
        if (useOnlyGreen) {
            cv::Mat split[3];
            cv::split(img, split);
            img = split[1];
        }
        paint = img.clone();
    }
    if (plotMarkers && paint.channels() == 1) {
        cv::Mat tmp[3] = {paint, paint, paint};
        cv::merge(tmp, 3, paint);
    }
    if (paint.depth() == CV_16U || paint.depth() == CV_16S) {
        std::cout << "Input image 16 bit, converting for painting to 8 bit." << std::endl;
        paint.convertTo(paint, CV_8UC3, 1.0 / 256.0);
    }
    std::cout << "Paint type: " << paint.type() << std::endl;
    std::cout << "Paint depth: " << paint.depth() << std::endl;
    std::cout << "Input image size of file " << input_file << ": " << img.size << std::endl;

    Marker::init();

    if (!read_cache_success) {
        detect(img, corners,use_rgb,0,10, effort, 3);
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


    std::vector<cv::Scalar> const color_circle = {
        cv::Scalar(255,255,255),
        cv::Scalar(255,0,0),
        cv::Scalar(0,255,0),
        cv::Scalar(0,0,255),
        cv::Scalar(255,255,0),
        cv::Scalar(0,255,255),
        cv::Scalar(255,0,255),
    };


    Mat gray;
    if (img.channels() != 1) {
        cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    }
    else {
        gray = img;
    }

    if (recursionDepth > 0) {
        std::cout << "Drawing sub-markers" << std::endl;
        double msize = 1.0;
        if (!read_submarkers_success) {
            //hdmarker::refine_recursive(gray, corners, submarkers, recursionDepth, &msize);
            //piecewiseRefinement(gray, corners, submarkers, recursion_depth, msize);
            refineRecursiveByPage(gray, corners, submarkers, recursionDepth, msize);
            FileStorage submarker_cache(submarkers_file, FileStorage::WRITE);
            submarker_cache << "corners" << "[";
            for (hdmarker::Corner const& c : submarkers) {
                submarker_cache << c;
            }
            submarker_cache << "]";
            submarker_cache.release();
        }


        if (plotMarkers) {
            cv::Mat paint2 = paint.clone();
            for (hdmarker::Corner const& c : submarkers) {
                circle(paint2, c.p, 3, Scalar(0,0,255,0), -1, LINE_AA);
            }
            imwrite(input_file + "-2.png", paint2);
        }

        if (submarkers.size() <= corners.size()) {
            std::cout << "Warning: Number of submarkers (" << std::to_string(submarkers.size())
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
#pragma omp critical
        for (size_t ii = 0; ii < corners.size(); ++ii) {
            if (0 != corners[ii].id.x) { // Checking the factor for x:
                const int current_factor = submarkers[ii].id.x / corners[ii].id.x;
                if (1 == cornerIdFactor) {
                    cornerIdFactor = current_factor;
                }
                if (current_factor != cornerIdFactor) {
                    throw std::runtime_error(std::string("Factor for x at corner #") + std::to_string(ii) +
                                             " (" + std::to_string(current_factor) + ") differs from first factor(" +
                                             std::to_string(cornerIdFactor) + "), aborting.");
                }
            }
            if (0 != corners[ii].id.y) { // Checking the factor for y:
                const int current_factor = submarkers[ii].id.y / corners[ii].id.y;
                if (1 == cornerIdFactor) {
                    cornerIdFactor = current_factor;
                }
                if (current_factor != cornerIdFactor) {
                    throw std::runtime_error(std::string("Factor for y at corner #") + std::to_string(ii) +
                                             " (" + std::to_string(current_factor) + ") differs from first factor(" +
                                             std::to_string(cornerIdFactor) + "), aborting.");
                }
            }
        }

    }

    std::cout << "Purging duplicate submarkers: " << submarkers.size() << " -> ";
    CornerStore c(submarkers);
    c.purgeDuplicates();
    submarkers = c.getCorners();
    std::cout << submarkers.size() << std::endl;

    if (plotMarkers) {
        //std::sort(submarkers.begin(), submarkers.end(), CornerIdSort());
        cv::Mat paint_submarkers = paint.clone();
        if (recursionDepth > 0) {
            int paint_sm_factor = 2;
            if (paint.cols < 3000 && paint.rows < 3000) {
                paint_sm_factor = 5;
            }
            cv::resize(paint_submarkers, paint_submarkers, cv::Size(), paint_sm_factor, paint_sm_factor, cv::INTER_NEAREST);
            for(size_t ii = 0; ii < submarkers.size(); ++ii) {
                Corner const& c = submarkers[ii];


                cv::Scalar const& font_color = color_circle[ii % color_circle.size()];
                std::string const text = to_string(c.id.x) + "/" + to_string(c.id.y) + "/" + to_string(c.page);
                circle(paint_submarkers, paint_sm_factor*c.p, 1, Scalar(0,0,0,0), 2);
                circle(paint_submarkers, paint_sm_factor*c.p, 1, Scalar(0,255,0,0));
                putText(paint_submarkers, text.c_str(), paint_sm_factor*c.p, FONT_HERSHEY_PLAIN, 1.2, Scalar(0,0,0,0), 2, cv::LINE_AA);
                putText(paint_submarkers, text.c_str(), paint_sm_factor*c.p, FONT_HERSHEY_PLAIN, 1.2, font_color, 1, cv::LINE_AA);
                std::string const text_page = to_string(c.page);
                double font_size = 2;
                if (c.id.x % cornerIdFactor == 0 && c.id.y % cornerIdFactor == 0) {
                    cv::Point2f const point_page = paint_sm_factor * (c.p + cv::Point2f(c.size/2 - font_size*5, c.size/2 - font_size*5));
                    putText(paint_submarkers, text_page.c_str(), point_page, FONT_HERSHEY_PLAIN, font_size, Scalar(0,0,0,0), 2, cv::LINE_AA);
                    putText(paint_submarkers, text_page.c_str(), point_page, FONT_HERSHEY_PLAIN, font_size, color_circle[c.page % color_circle.size()], 1, cv::LINE_AA);
                }
            }
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
            cv::Point2f const point_page = c.p + cv::Point2f(c.size/2 - font_size*5, c.size/2 - font_size*5);
            putText(paint, text_page.c_str(), point_page, FONT_HERSHEY_PLAIN, font_size, Scalar(0,0,0,0), 2, cv::LINE_AA);
            putText(paint, text_page.c_str(), point_page, FONT_HERSHEY_PLAIN, font_size, color_circle[c.page % color_circle.size()], 1, cv::LINE_AA);
        }
        imwrite(input_file + "-1.png", paint);
    }


    if (recursionDepth > 0) {
        return submarkers;
    }

    return corners;

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
                if (cv::norm(a.p-b.p) < (a.size + b.size)/20) {
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

size_t CornerStore::size() const {
    return corners.size();
}

const Corner &CornerStore::get(size_t index) const {
    if (index >= corners.size()) {
        throw std::out_of_range(std::string("Index ") + to_string(index) + " too large for current size of corners vector (" + to_string(corners.size()) + ")");
    }
    return corners[index];
}

void CornerStore::push_back(const Corner x) {
    corners.push_back(x);
    idx_tree->addPoints(corners.size()-1, corners.size()-1);
    pos_tree->addPoints(corners.size()-1, corners.size()-1);
}

void CornerStore::push_conditional(const Corner x) {
    if (!hasID(x)) {
        push_back(x);
    }
}

void CornerStore::add(const std::vector<Corner> &vec) {
    if (vec.empty()) {
        return;
    }
    corners.insert(corners.end(), vec.begin(), vec.end());
    idx_tree->addPoints(corners.size() - vec.size(), corners.size()-1);
    pos_tree->addPoints(corners.size() - vec.size(), corners.size()-1);
}

void CornerStore::getPoints(
        std::vector<Point2f> &imagePoints,
        std::vector<Point3f> &objectPoints,
        hdcalib::Calib const& calib) const {
    imagePoints.clear();
    imagePoints.reserve(size());

    objectPoints.clear();
    objectPoints.reserve(size());
    for (size_t ii = 0; ii < size(); ++ii) {
        imagePoints.push_back(get(ii).p);
        objectPoints.push_back(calib.getInitial3DCoord(get(ii)));
    }
}

CornerStore::CornerStore() :
    idx_adapt(*this),
    pos_adapt(*this),
    idx_tree(new CornerIndexTree(
                 3 /*dim*/,
                 idx_adapt,
                 nanoflann::KDTreeSingleIndexAdaptorParams(16 /* max leaf */)
                 )),
    pos_tree(new CornerPositionTree (
                 2 /*dim*/,
                 pos_adapt,
                 nanoflann::KDTreeSingleIndexAdaptorParams(16 /* max leaf */)
                 )) {

}

CornerStore::CornerStore(const CornerStore &c) :
    idx_adapt(*this),
    pos_adapt(*this),
    idx_tree(new CornerIndexTree(
                 3 /*dim*/,
                 idx_adapt,
                 nanoflann::KDTreeSingleIndexAdaptorParams(16 /* max leaf */)
                 )),
    pos_tree(new CornerPositionTree (
                 2 /*dim*/,
                 pos_adapt,
                 nanoflann::KDTreeSingleIndexAdaptorParams(16 /* max leaf */)
                 )) {
    replaceCorners(c.getCorners());
}

CornerStore::CornerStore(const std::vector<Corner> &corners) :
    idx_adapt(*this),
    pos_adapt(*this),
    idx_tree(new CornerIndexTree(
                 3 /*dim*/,
                 idx_adapt,
                 nanoflann::KDTreeSingleIndexAdaptorParams(16 /* max leaf */)
                 )),
    pos_tree(new CornerPositionTree (
                 2 /*dim*/,
                 pos_adapt,
                 nanoflann::KDTreeSingleIndexAdaptorParams(16 /* max leaf */)
                 )) {
    replaceCorners(corners);
}

CornerStore &CornerStore::operator=(const CornerStore &other) {
    if (this != &other) { // protect against invalid self-assignment
        replaceCorners(other.getCorners());
    }
    // by convention, always return *this
    return *this;
}

void CornerStore::clean(int cornerIdFactor) {
    //purge32();
    purgeDuplicates();
    purgeUnlikely(cornerIdFactor);
}

void CornerStore::intersect(const CornerStore &b) {
    bool found_delete = false;
    std::vector<hdmarker::Corner> replacement;
    for (size_t ii = 0; ii < size(); ++ii) {
        if (b.hasID(get(ii))) {
            replacement.push_back(get(ii));
        }
        else {
            found_delete = true;
        }
    }
    if (found_delete) {
        replaceCorners(replacement);
    }
}

void CornerStore::intersect(CornerStore &a, CornerStore &b) {
    a.intersect(b);
    if (a.size() != b.size()) {
        b.intersect(a);
    }
}

void CornerStore::difference(const CornerStore &subtrahend) {
    bool found_delete = false;
    std::vector<hdmarker::Corner> replacement;
    for (size_t ii = 0; ii < size(); ++ii) {
        if (subtrahend.hasID(get(ii))) {
            found_delete = true;
        }
        else {
            replacement.push_back(get(ii));
        }
    }
    if (found_delete) {
        replaceCorners(replacement);
    }
}

void CornerStore::replaceCorners(const std::vector<Corner> &_corners) {
    corners = _corners;
    {
        std::shared_ptr<CornerIndexTree> idx_tree_replacement(new CornerIndexTree(
                                                                  3 /*dim*/,
                                                                  idx_adapt,
                                                                  nanoflann::KDTreeSingleIndexAdaptorParams(16 /* max leaf */)));
        idx_tree.swap(idx_tree_replacement);
    }
    {
        std::shared_ptr<CornerPositionTree> pos_tree_replacement(new CornerPositionTree(
                                                                     2 /*dim*/,
                                                                     pos_adapt,
                                                                     nanoflann::KDTreeSingleIndexAdaptorParams(16 /* max leaf */)));
        pos_tree.swap(pos_tree_replacement);
    }
    if (corners.size() > 0) {
        idx_tree->addPoints(0, corners.size() - 1);
        pos_tree->addPoints(0, corners.size() - 1);
    }
}

std::vector<Corner> CornerStore::getCorners() const {
    return corners;
}

std::vector<Corner> CornerStore::findByID(const Corner &ref, const size_t num_results) const {
    std::vector<hdmarker::Corner> result;
    double query_pt[3] = {
        static_cast<double>(ref.id.x),
        static_cast<double>(ref.id.y),
        static_cast<double>(ref.page)
    };

    // do a knn search
    std::unique_ptr<size_t[]> res_indices(new size_t[num_results]);
    std::unique_ptr<double[]> res_dist_sqr( new double[num_results]);
    nanoflann::KNNResultSet<double> resultSet(num_results);
    resultSet.init(res_indices.get(), res_dist_sqr.get());
    idx_tree->findNeighbors(resultSet, query_pt, nanoflann::SearchParams(10));

    for (size_t ii = 0; ii < resultSet.size(); ++ii) {
        result.push_back(corners[res_indices[ii]]);
    }

    return result;
}

std::vector<Corner> CornerStore::findByPos(const Corner &ref, const size_t num_results) {
    return findByPos(ref.p.x, ref.p.y, num_results);
}

std::vector<Corner> CornerStore::findByPos(const double x, const double y, const size_t num_results) {
    std::vector<hdmarker::Corner> result;
    double query_pt[2] = {
        static_cast<double>(x),
        static_cast<double>(y)
    };

    // do a knn search
    std::unique_ptr<size_t[]> res_indices(new size_t[num_results]);
    std::unique_ptr<double[]> res_dist_sqr( new double[num_results]);
    for (size_t ii = 0; ii < num_results; ++ii) {
        res_indices[ii] = 0;
        res_dist_sqr[ii] = std::numeric_limits<double>::max();
    }
    nanoflann::KNNResultSet<double> resultSet(num_results);
    resultSet.init(res_indices.get(), res_dist_sqr.get());
    pos_tree->findNeighbors(resultSet, query_pt, nanoflann::SearchParams(10));

    for (size_t ii = 0; ii < resultSet.size(); ++ii) {
        size_t const index = res_indices[ii];
        hdmarker::Corner const& c = corners[index];
        result.push_back(c);
    }

    return result;
}

bool CornerStore::purgeUnlikely(int cornerIdFactor) {
    std::vector<hdmarker::Corner> keep;
    keep.reserve(size());

    for (size_t ii = 0; ii < size(); ++ii) {
        hdmarker::Corner const& candidate = get(ii);
        std::vector<hdmarker::Corner> const res = findByPos(candidate, 5);
        size_t neighbours = 0;
        for (hdmarker::Corner const& neighbour : res) {
            if (neighbour.page != candidate.page) {
                continue;
            }
            if (neighbour.id == candidate.id) {
                continue;
            }
            cv::Point2i residual = candidate.id - neighbour.id;
            if (std::abs(residual.x) <= cornerIdFactor && std::abs(residual.y) <= cornerIdFactor) {
                neighbours++;
                if (neighbours > 1) {
                    keep.push_back(candidate);
                    break;
                }
            }
        }
    }

    if (size() != keep.size()) {
        replaceCorners(keep);
        return true;
    }
    return false;
}

bool CornerStore::purgeDuplicates() {
    std::vector<hdmarker::Corner> keep;
    keep.reserve(size());

    for (size_t ii = 0; ii < size(); ++ii) {
        hdmarker::Corner const& candidate = get(ii);
        size_t const num_results = 16;
        double query_pt[2] = {
            static_cast<double>(candidate.p.x),
            static_cast<double>(candidate.p.y)
        };

        // do a knn search
        std::unique_ptr<size_t[]> res_indices(new size_t[num_results]);
        std::unique_ptr<double[]> res_dist_sqr( new double[num_results]);
        nanoflann::KNNResultSet<double> resultSet(num_results);
        resultSet.init(res_indices.get(), res_dist_sqr.get());
        pos_tree->findNeighbors(resultSet, query_pt, nanoflann::SearchParams(10));

        bool is_duplicate = false;
        for (size_t jj = 0; jj < resultSet.size(); ++jj) {
            if (ii <= res_indices[jj]) {
                continue;
            }
            hdmarker::Corner const& b = get(res_indices[jj]);
            cv::Point2f residual = candidate.p - b.p;
            double const dist = Calib::distance(candidate.p, b.p);
            if (dist < (candidate.size + b.size)/20 && dist < 5) {
                is_duplicate = true;
                break;
            }
            break;
        }
        if (!is_duplicate) {
            keep.push_back(candidate);
        }
    }

    if (size() != keep.size()) {
        replaceCorners(keep);
        return true;
    }
    return false;
}

bool CornerStore::purge32() {
    std::vector<hdmarker::Corner> res;
    res.reserve(size());
    for (size_t ii = 0; ii < size(); ++ii) {
        hdmarker::Corner const& candidate = get(ii);
        if (candidate.id.x != 32 && candidate.id.y != 32) {
            res.push_back(candidate);
            continue;
        }
        auto const search_res = findByPos(candidate, 2);
        if (search_res.size() < 2) {
            res.push_back(candidate);
            continue;
        }
        hdmarker::Corner const& second = search_res[1];
        cv::Point2f const diff = candidate.p - second.p;
        double const dist = std::sqrt(diff.dot(diff));
        if (dist > (candidate.size + second.size)/20) {
            res.push_back(candidate);
        }
    }
    if (res.size() != size()) {
        replaceCorners(res);
        return true;
    }
    return false;
}


double Calib::openCVCalib() {
    prepareCalibration();

    int flags = 0;
    //flags |= CALIB_FIX_PRINCIPAL_POINT;
    //flags |= CALIB_FIX_ASPECT_RATIO;
    flags |= CALIB_RATIONAL_MODEL;
    flags |= CALIB_THIN_PRISM_MODEL;
    flags |= CALIB_TILTED_MODEL;

    if (imageSize.height == 5320 && imageSize.width == 7968) { // Hacky detection of my Sony setup.
        cameraMatrix = (Mat_<double>(3,3) << 12937, 0, 4083, 0, 12978, 2636, 0, 0, 1);
        flags |= CALIB_USE_INTRINSIC_GUESS;
    }
    std::cout << "Initial camera matrix: " << std::endl << cameraMatrix << std::endl;

    double result_err = cv::calibrateCamera (
                objectPoints,
                imagePoints,
                imageSize,
                cameraMatrix,
                distCoeffs,
                rvecs,
                tvecs,
                stdDevIntrinsics,
                stdDevExtrinsics,
                perViewErrors,
                flags
                );

    std::cout << "RMSE: " << result_err << std::endl;
    std::cout << "Camera Matrix: " << std::endl << cameraMatrix << std::endl;
    std::cout << "distCoeffs: " << std::endl << distCoeffs << std::endl;
    std::cout << "stdDevIntrinsics: " << std::endl << stdDevIntrinsics << std::endl;
    std::cout << "stdDevExtrinsics: " << std::endl << stdDevExtrinsics << std::endl;
    std::cout << "perViewErrors: " << std::endl << perViewErrors << std::endl;

    std::cout << "rvecs: " << std::endl;
    for (auto const& rvec: rvecs) {
        std::cout << rvec << std::endl;
    }

    std::cout << "tvecs: " << std::endl;
    for (auto const& tvec: tvecs) {
        std::cout << tvec << std::endl;
    }

    cv::calibrationMatrixValues (
                cameraMatrix,
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
    std::cout << "calibrationMatrixValues: " << std::endl
              << "fovx: " << fovx << std::endl
              << "fovy: " << fovy << std::endl
              << "focalLength: " << focalLength << std::endl
              << "principalPoint: " << principalPoint << std::endl
              << "aspectRatio: " << aspectRatio << std::endl
              << "input image size: " << imageSize << std::endl
              << "pixel size (um): " << pixel_size * 1000 << std::endl << std::endl;

    cv::Point2d principal_point_offset = principalPoint - cv::Point2d(apertureWidth/2, apertureHeight/2);
    std::cout << "principal point offset: " << principal_point_offset << "mm; ~" << principal_point_offset/pixel_size << "px" << std::endl;

    std::cout << "focal length factor: " << cameraMatrix(0,0) / focalLength << std::endl;

    return result_err;
}

double Calib::CeresCalib() {

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
    options.max_num_iterations = 150;
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
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.max_num_iterations = 150;
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

void Calib::setPlotMarkers(bool plot) {
    plotMarkers = plot;
}

void Calib::setImageSize(const Mat &img) {
    imageSize = cv::Size(img.size());
    resolutionKnown = true;
}

bool CornerStore::hasID(const Corner &ref) const {
    double query_pt[3] = {
        static_cast<double>(ref.id.x),
        static_cast<double>(ref.id.y),
        static_cast<double>(ref.page)
    };
    size_t res_index;
    double res_dist_sqr;
    nanoflann::KNNResultSet<double> resultSet(1);
    resultSet.init(&res_index, &res_dist_sqr);
    idx_tree->findNeighbors(resultSet, query_pt, nanoflann::SearchParams(10));

    if (resultSet.size() < 1) {
        return false;
    }
    hdmarker::Corner const& result = corners[res_index];
    return result.id == ref.id && result.page == ref.page;
}

CornerIndexAdaptor::CornerIndexAdaptor(CornerStore &ref) : store(&ref){

}

size_t CornerIndexAdaptor::kdtree_get_point_count() const {
    return store->size();
}

int CornerIndexAdaptor::kdtree_get_pt(const size_t idx, int dim) const {
    hdmarker::Corner const& c = store->get(idx);
    if (0 == dim) {
        return c.id.x;
    }
    if (1 == dim) {
        return c.id.y;
    }
    if (2 == dim) {
        return c.page;
    }
    throw std::out_of_range("Dimension number " + to_string(dim) + " out of range (0-2)");
}

CornerPositionAdaptor::CornerPositionAdaptor(CornerStore &ref) : store(&ref) {

}

size_t CornerPositionAdaptor::kdtree_get_point_count() const {
    return store->size();
}

int CornerPositionAdaptor::kdtree_get_pt(const size_t idx, int dim) const {
    hdmarker::Corner const& c = store->get(idx);
    if (0 == dim) {
        return c.p.x;
    }
    if (1 == dim) {
        return c.p.y;
    }
    throw std::out_of_range(std::string("Requested dimension ") + to_string(dim) + " out of range (0-1)");
}



void Calib::plotReprojectionErrors(const size_t image_index,
                                   MarkerMap &residuals_by_marker,
                                   const std::string prefix,
                                   const std::string suffix) {
    std::string const& filename = imageFiles[image_index];

    std::stringstream plot_command;
    gnuplotio::Gnuplot plot;

    CornerStore const& store = data[filename];

    std::string plot_name = prefix + filename + ".marker-residuals";

    std::vector<double> errors;

    runningstats::Histogram error_hist(.1);

    runningstats::RunningCovariance proj_x, proj_y;

    runningstats::QuantileStats<double> error_stats;

    std::vector<cv::Point2d> markers, reprojections;

    getReprojections(image_index, markers, reprojections);

    std::vector<std::vector<double> > data;
    for (size_t ii = 0; ii < markers.size(); ++ii) {
        double const error = distance(markers[ii], reprojections[ii]);
        data.push_back({markers[ii].x, markers[ii].y,
                        reprojections[ii].x, reprojections[ii].y,
                        error});
        proj_x.push(markers[ii].x, reprojections[ii].x);
        proj_y.push(markers[ii].y, reprojections[ii].y);
        auto const id = getSimpleId(store.get(ii));
        residuals_by_marker[id].push_back(std::make_pair(markers[ii], reprojections[ii]));
        errors.push_back(error);
        error_stats.push(error);
        error_hist.push(error);
    }


    std::cout << "Error stats for image " << filename << ": "
              << std::endl << error_hist.printBoth() << ", quantiles for .25, .5, .75, .9, .95: "
              << error_stats.getQuantile(.25) << ", "
              << error_stats.getQuantile(.5) << ", "
              << error_stats.getQuantile(.75) << ", "
              << error_stats.getQuantile(.9) << ", "
              << error_stats.getQuantile(.95) << ", "
              << std::endl;

    std::cout << "Covariance between marker x-values and reprojection x-values: " << proj_x.getCorr() << std::endl;
    std::cout << "Covariance between marker y-values and reprojection y-values: " << proj_y.getCorr() << std::endl;

    std::cout << std::endl;

    std::sort(errors.begin(), errors.end());

    plot_command << std::setprecision(16);
    plot_command << "set term svg enhanced background rgb \"white\";\n"
                 << "set output \"" << plot_name << ".residuals." << suffix << ".svg\";\n"
                 << "set title 'Reprojection Residuals';\n"
                 << "plot " << plot.file1d(data, plot_name + ".residuals." + suffix + ".data")
                 << " u ($1-$3):($2-$4) w p notitle;\n"
                 << "set output \"" << plot_name << ".residuals-log." << suffix << ".svg\";\n"
                 << "set title 'Reprojection Residuals';\n"
                 << "set logscale xy;\n"
                 << "plot " << plot.file1d(data, plot_name + ".residuals." + suffix + ".data")
                 << " u (abs($1-$3)):(abs($2-$4)) w p notitle;\n"
                 << "reset;\n"
                 << "set output \"" << plot_name + ".vectors." << suffix << ".svg\";\n"
                 << "set title 'Reprojection Residuals';\n"
                 << "plot " << plot.file1d(data, plot_name + ".residuals." + suffix + ".data")
                 << " u 1:2:($3-$1):($4-$2) w vectors notitle;\n"
                 << "reset;\n"
                 << "set output \"" << plot_name + ".vectors." << suffix << ".2.svg\";\n"
                 << "set title 'Reprojection Residuals';\n"
                 << "plot " << plot.file1d(data, plot_name + ".residuals." + suffix + ".data")
                 << " u 3:4:($1-$3):($2-$4) w vectors notitle;\n"
                 << "reset;\n"
                 << "set key out horiz;\n"
                 << "set output \"" << plot_name + ".images." << suffix << ".svg\";\n"
                 << "set title 'Reprojection vs. original';\n"
                 << "plot " << plot.file1d(data, plot_name + ".residuals." + suffix + ".data")
                 << " u 1:2 w points title 'detected', \"" << plot_name + ".residuals." + suffix << ".data\" u 3:4 w points title 'reprojected';\n"
                 << "set output \"" << plot_name + ".error-dist." << suffix << ".svg\";\n"
                 << "set title 'CDF of the Reprojection Error';\n"
                 << "set xlabel 'error';\n"
                 << "set ylabel 'CDF';\n"
                 << "plot " << plot.file1d(errors, plot_name + ".errors." + suffix + ".data") << " u 1:($0/" << errors.size()-1 << ") w l notitle;\n"
                 << "set logscale x;\n"
                 << "set output \"" << plot_name + ".error-dist-log." << suffix << ".svg\";\n"
                 << "replot;\n"
                 << "reset;\n"
                 << "set output \"" << plot_name + ".error-hist." << suffix << ".svg\";\n"
                 << "set title 'Reprojection Error Histogram';\n"
                 << "set xlabel 'error';\n"
                 << "set ylabel 'absolute frequency';\n"
                 << "plot " << plot.file1d(error_hist.getAbsoluteHist(), plot_name + ".errors-hist." + suffix + ".data")
                 << " w boxes notitle;\n"
                 << "set output \"" << plot_name + ".error-hist-log." << suffix << ".svg\";\n"
                 << "set logscale xy;\n"
                 << "plot " << plot.file1d(error_hist.getAbsoluteHist(), plot_name + ".errors-hist." + suffix + ".data")
                 << "w boxes notitle;\n";

    plot << plot_command.str();

    std::ofstream out(plot_name + "." + suffix + ".gpl");
    out << plot_command.str();

}

void Calib::plotErrorsByMarker(
        const Calib::MarkerMap &map,
        const string prefix,
        const string suffix) {

    for (auto const& it : map) {
        if (it.second.size() < 2) {
            continue;
        }
        std::stringstream _id;
        _id << it.first;
        auto const id = _id.str();

        std::vector<std::vector<float> > local_data;
        for (auto const& d : it.second) {
            local_data.push_back({d.first.x, d.first.y, d.second.x, d.second.y});
        }

        std::string plot_name = "markers." + id;
        gnuplotio::Gnuplot plot;
        std::stringstream plot_command;
        plot_command << std::setprecision(16);
        plot_command << "set term svg enhanced background rgb \"white\";\n"
                     << "set output \"" << plot_name << ".residuals." << suffix << ".svg\";\n"
                     << "set title 'Reprojection Residuals for marker " << id << "';\n"
                     << "plot " << plot.file1d(local_data, plot_name + ".residuals." + suffix + ".data")
                     << " u ($1-$3):($2-$4) w p notitle;\n";


        plot << plot_command.str();

        std::ofstream out(plot_name + "." + suffix + ".gpl");
        out << plot_command.str();
    }
}

void Calib::plotResidualsByMarkerStats(
        const Calib::MarkerMap &map,
        const string prefix,
        const string suffix) {

    std::vector<std::pair<double, double> > mean_residuals_by_marker;
    int max_x = 0, max_y = 0;
    for (auto const& it : map) {
        cv::Point3f const f_coord = getInitial3DCoord(it.first);
        max_x = std::max(max_x, 1+int(std::ceil(f_coord.x)));
        max_y = std::max(max_y, 1+int(std::ceil(f_coord.y)));
    }
    cv::Mat_<cv::Vec2f> residuals(max_y, max_x, cv::Vec2f(0,0));
    cv::Mat_<uint8_t> errors(max_y, max_x, uint8_t(0));
    for (auto const& it : map) {
        const cv::Point2f mean_res = meanResidual(it.second);
        mean_residuals_by_marker.push_back({mean_res.x, mean_res.y});
        cv::Point3f const f_coord = getInitial3DCoord(it.first);
        residuals(int(f_coord.y), int(f_coord.x)) = cv::Vec2f(mean_res);
        errors(int(f_coord.y), int(f_coord.x)) = cv::saturate_cast<uint8_t>(255*std::sqrt(mean_res.dot(mean_res)));
    }

    std::stringstream plot_command;
    std::string plot_name = prefix + "residuals-by-marker";

    cv::writeOpticalFlow(plot_name + "." + suffix + ".flo", residuals);
    cv::imwrite(plot_name + ".errors." + suffix + ".png", errors);

    gnuplotio::Gnuplot plot;

    plot_command << "set term svg enhanced background rgb \"white\";\n"
                 << "set output \"" << plot_name << "." << suffix << ".svg\";\n"
                 << "set title 'Mean reprojection residuals of each marker';\n"
                 << "set size ratio -1;\n"
                 << "plot " << plot.file1d(mean_residuals_by_marker, plot_name + "." + suffix + ".data")
                 << " u 1:2 w p notitle;\n";

    plot << plot_command.str();
    std::ofstream out(plot_name + ".gpl");
    out << plot_command.str();
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

void Calib::white_balance_inplace(Mat &mat, const Point3f white) {
    float const min_val = std::min(white.x, std::min(white.y, white.z));
    float const max_val = std::max(white.x, std::max(white.y, white.z));
    float const f_b = min_val / white.x;
    float const f_g = min_val / white.y;
    float const f_r = min_val / white.z;
    for (int ii = 0; ii < mat.rows; ++ii) {
        unsigned short * row = mat.ptr<unsigned short>(ii);
        for (int jj = 0; jj < mat.cols; ++jj) {
            switch (color(ii, jj)) {
            case 'R': row[jj] = cv::saturate_cast<unsigned short>(std::round(row[jj] * f_r)); break;
            case 'G': row[jj] = cv::saturate_cast<unsigned short>(std::round(row[jj] * f_g)); break;
            case 'B': row[jj] = cv::saturate_cast<unsigned short>(std::round(row[jj] * f_b)); break;
            default: throw std::runtime_error("Color value not R,G,B");
            }
        }
    }
}

void Calib::plotReprojectionErrors(const string prefix, const string suffix) {
    MarkerMap residuals_by_marker;
    for (size_t ii = 0; ii < imagePoints.size(); ++ii) {
        plotReprojectionErrors(ii, residuals_by_marker, prefix, suffix);
    }
    //plotErrorsByMarker(residuals_by_marker);
    plotResidualsByMarkerStats(residuals_by_marker, prefix, suffix);
}

void Calib::findOutliers(
        const double threshold,
        const size_t image_index,
        std::vector<hdmarker::Corner> & outliers) {
    std::vector<cv::Point2d> markers, projections;
    getReprojections(image_index, markers, projections);

    for (size_t ii = 0; ii < markers.size(); ++ii) {
        double const error = distance(markers[ii], projections[ii]);
        if (error < threshold) {
            continue;
        }
        hdmarker::Corner c = data[imageFiles[image_index]].get(ii);
        outliers.push_back(c);
        if (verbose) {
            std::cout << "found outlier in image " << imageFiles[image_index]
                         << ": id " << c.id << ", " << c.page << ", marker: "
                         << markers[ii] << ", proj: " << projections[ii]
                            << ", dist: " << error << std::endl;
        }
    }
}

Vec3d Calib::get3DPoint(const Corner &c, const Mat &_rvec, const Mat &_tvec) {
    cv::Mat_<double> rvec(_rvec);
    cv::Mat_<double> tvec(_tvec);
    cv::Point3f _src = getInitial3DCoord(c);
    _src += objectPointCorrections[getSimpleId(c)];
    double src[3] = {_src.x, _src.y, _src.z};
    double rot[9];
    double rvec_data[3] = {rvec(0), rvec(1), rvec(2)};
    double tvec_data[3] = {tvec(0), tvec(1), tvec(2)};
    rot_vec2mat(rvec_data, rot);
    double _result[3] = {0,0,0};
    get3DPoint(src, _result, rot, tvec_data);

    return cv::Vec3d(_result[0], _result[1], _result[2]);
}

void Calib::getRectificationRotation(const size_t rows, const size_t cols, const std::vector<string> &images, Vec3d &rect_rot) {

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

template<class Point>
double Calib::distance(const Point a, const Point b) {
    Point residual = a-b;
    return std::sqrt(residual.dot(residual));
}

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

template<class C>
bool cmpSimpleIndex3<C>::operator()(const C &a, const C &b) const {
    if (a.z < b.z) return true;
    if (a.z > b.z) return false;
    if (a.y < b.y) return true;
    if (a.y > b.y) return false;
    return a.x < b.x;
}

void write(FileStorage &fs, const string &, const Calib &x) {
    x.write(fs);
}

void read(const FileNode &node, Calib &x, const Calib &default_value) {
    if(node.empty()) {
        throw std::runtime_error("Could not recover calibration, file storage is empty.");
    }
    else
        x.read(node);
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

}
