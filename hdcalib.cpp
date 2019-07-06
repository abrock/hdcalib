#include "hdcalib.h"

#include <ceres/ceres.h>
#include <opencv2/optflow.hpp>
#include <opencv2/video/tracking.hpp>
#include <runningstats/runningstats.h>
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

void Calib::removeOutliers(const double threshold) {
    prepareCalibration();
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
    msg << "Outlier percentage by image:" << std::endl;
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
    invalidateCache();
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

Calib::Calib() {
    std::cout << "Number of concurrent threads: " << threads << std::endl;
}

void Calib::invalidateCache() {
    preparedCalib = false;
    preparedOpenCVCalib = false;
    imagePoints.clear();
    objectPoints.clear();
}

void Calib::purgeInvalidPages() {
    invalidateCache();
    for (auto& it : data) {
        std::vector<hdmarker::Corner> cleaned = purgeInvalidPages(it.second.getCorners(), validPages);
        if (cleaned.size() < it.second.size()) {
            if (verbose) {
                std::cout << "In image " << it.first << " removed " << it.second.size() - cleaned.size() << " out of " << it.second.size() << " corners" << std::endl;
            }
            it.second.replaceCorners(cleaned);
        }
    }
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

void Calib::setRecursionDepth(int _recursionDepth) {
    recursionDepth = _recursionDepth;
    cornerIdFactor = 1;
    if (recursionDepth > 0) {
        cornerIdFactor = 10;
        for (size_t ii = 1; ii < recursionDepth; ++ii) {
            cornerIdFactor *= 5;
        }
    }
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
    for (std::pair<std::string, CornerStore> const& it : data) {
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

CornerStore Calib::get(const string filename) const {
    auto const it = data.find(filename);
    if (it != data.end()) {
        return it->second;
    }
    throw std::runtime_error(std::string("File ") + filename + " not found in data.");
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

            cvtColor(img, img, COLOR_BayerBG2BGR); // RG BG GB GR
        }
        else {
            img = cv::imread(input_file);
            setImageSize(img);
            std::cout << "Input file " << input_file << " image size: " << img.size() << std::endl;
        }
        if (useOnlyGreen) {
            if (img.channels() > 1) {
                cv::Mat split[3];
                cv::split(img, split);
                img = split[1];
            }
        }
        if (img.channels() == 1 && (img.depth() == CV_16U || img.depth() == CV_16S)) {
            std::cout << "Input image is 1 channel, 16 bit, converting for painting to 8 bit." << std::endl;
            img.convertTo(img, CV_8UC1, 1.0 / 256.0);
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

    if (img.depth() == CV_16U || img.depth() == CV_16S) {
        std::cout << "Input image for marker detection 16 bit, converting for painting to 8 bit." << std::endl;
        paint.convertTo(paint, CV_8UC1, 1.0 / 256.0);
    }
    if (!read_cache_success) {
        detect(img, corners, use_rgb, 0, 10, effort);
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


                cv::Point2f local_shift(0,0);
                if (c.id.x % 10 != 0 && c.id.y % 10 == 0) {
                    if (c.id.x % 10 == 3 || c.id.x % 10 == 7) {
                        local_shift.y = 16;
                    }
                }


                cv::Scalar const& font_color = color_circle[ii % color_circle.size()];
                std::string const text = to_string(c.id.x) + "/" + to_string(c.id.y) + "/" + to_string(c.page);
                circle(paint_submarkers, paint_sm_factor*c.p, 1, Scalar(0,0,0,0), 2);
                circle(paint_submarkers, paint_sm_factor*c.p, 1, Scalar(0,255,0,0));
                putText(paint_submarkers, text.c_str(), local_shift+paint_sm_factor*c.p, FONT_HERSHEY_PLAIN, .9, Scalar(0,0,0,0), 2, cv::LINE_AA);
                putText(paint_submarkers, text.c_str(), local_shift+paint_sm_factor*c.p, FONT_HERSHEY_PLAIN, .9, font_color, 1, cv::LINE_AA);
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

double Calib::openCVCalib() {
    prepareOpenCVCalibration();

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

void Calib::setPlotMarkers(bool plot) {
    plotMarkers = plot;
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
        if (verbose && verbose2) {
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

template<class Point>
double Calib::distance(const Point a, const Point b) {
    Point residual = a-b;
    return std::sqrt(residual.dot(residual));
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

}
