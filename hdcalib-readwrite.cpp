#include "hdcalib.h"
#include <hdmarker/subpattern3.hpp>
#include <runningstats/runningstats.h>

#include <ParallelTime/paralleltime.h>

namespace hdcalib {

template<class T, class T1, class T2>
void Calib::insertSorted(std::vector<T>& a, std::vector<T1>& b, std::vector<T2>& c) {
    if (a.size() != b.size() || a.size() != c.size()) {
        throw std::runtime_error(std::string("Sizes of arrays do not match: ")
                                 + std::to_string(a.size()) + ", "
                                 + std::to_string(b.size()) + ", "
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

void Calib::addInputImage(const string filename, const std::vector<Corner> &corners) {
    if (hasCalibration) {
        addInputImageAfterwards(filename, corners);
        return;
    }
    invalidateCache();
    CornerStore & ref = data[filename];
    ref.replaceCorners(corners);
    ref.clean(cornerIdFactor);
    imageFiles.push_back(filename);
    std::sort(imageFiles.begin(), imageFiles.end());
}

void Calib::addInputImageAfterwards(const string filename, const std::vector<Corner> &corners) {
    if (hasFile(filename)) {
        clog::L(__func__, 2) << "Not adding duplicate image " << filename << std::endl;
        return;
    }
    clog::L(__func__, 2) << "Adding image " << filename << "..." << std::flush;
    invalidateCache();

    ParallelTime t;
    std::vector<std::string> const old_imageFiles = imageFiles;

    clog::L(__func__, 2) << "replacing corners..." << std::flush;
    t.start();
    CornerStore & ref = data[filename];
    ref.replaceCorners(validPages.empty() ? corners : purgeInvalidPages(corners, validPages));
    clog::L(__func__, 2) << "replacing corners: " << t.print() << std::endl;
    t.start();
    ref.clean(cornerIdFactor);
    clog::L(__func__, 2) << "cleaning corners: " << t.print() << std::endl;
    t.start();

    clog::L(__func__, 2) << "preparing calibration..." << std::flush;
    prepareOpenCVCalibration();
    clog::L(__func__, 2) << "preparing calibration: " << t.print() << std::endl;


    std::vector<CalibResult* > calibs;
    for (auto& it : calibrations) {
        CalibResult & current_res = it.second;
        current_res.rvecs.push_back(cv::Mat());
        current_res.tvecs.push_back(cv::Mat());

        current_res.imageFiles = old_imageFiles;
        current_res.imageFiles.push_back(filename);
        insertSorted(current_res.imageFiles, current_res.rvecs, current_res.tvecs);
        calibs.push_back(&it.second);
    }
    size_t index = 0;
    if (!calibs.empty()) {
        for (; index < calibs.front()->imageFiles.size(); ++index) {
            if (filename == calibs.front()->imageFiles[index]) {
                break;
            }
        }
    }

    t.start();
    clog::L(__func__, 2) << "solvePnP..." << std::endl;
//#pragma omp parallel for schedule(dynamic)
    for (size_t ii = 0; ii < calibs.size(); ++ii) {
        CalibResult & current_res = *(calibs[ii]);
        bool const success = cv::solvePnP (
                    objectPoints[index],
                    imagePoints[index],
                    current_res.cameraMatrix,
                    current_res.distCoeffs,
                    current_res.rvecs[index],
                    current_res.tvecs[index]);
        ignore_unused(success);
    }
    clog::L(__func__, 2) << "solvePnP: " << t.print() << std::endl;
}

void Calib::addInputImage(const string filename, const std::vector<Corner> &corners, cv::Mat const& rvec, cv::Mat const& tvec) {
    if (hasFile(filename)) {
        return;
    }
    invalidateCache();

    CornerStore & ref = data[filename];
    ref.replaceCorners(validPages.empty() ? corners : purgeInvalidPages(corners, validPages));
    ref.clean(cornerIdFactor);
    for (auto& it : calibrations) {
        it.second.rvecs.push_back(rvec);
        it.second.tvecs.push_back(tvec);
    }
}

void Calib::removeInputImage(const string filename) {
    int const index = getImageIndex(filename);
    if (index < 0) {
        return;
    }
    prepareCalibration();
    assert(size_t(index) < data.size());
    assert(data.size() == imageFiles.size());
    assert(data.size() == objectPoints.size());

    imageFiles.erase(imageFiles.begin() + index);
    objectPoints.erase(objectPoints.begin() + index);

    auto const it = data.find(filename);
    assert(it != data.end());
    data.erase(it);


    for (auto& it : calibrations) {
        it.second.rvecs.erase(it.second.rvecs.begin() + index);
        it.second.tvecs.erase(it.second.tvecs.begin() + index);
    }

    assert(size_t(index) < data.size());
    assert(data.size() == imageFiles.size());
    assert(data.size() == objectPoints.size());

}

int Calib::getImageIndex(const string& filename) {
    int result = 0;
    for (auto const& it : data) {
        if (filename == it.first) {
            return result;
        }
        result++;
    }
    return -1;
}

void Calib::addInputImage(const string filename, const CornerStore &corners) {
    invalidateCache();

    CornerStore & ref = data[filename];
    ref = corners;
    ref.clean(cornerIdFactor);
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

Mat Calib::read_raw(const string &filename) {
    LibRaw RawProcessor;

    auto& S = RawProcessor.imgdata.sizes;
    //auto& OUT = RawProcessor.imgdata.params;

    int ret;
    if ((ret = RawProcessor.open_file(filename.c_str())) != LIBRAW_SUCCESS) {
        throw std::runtime_error(std::string("Cannot open file ") + filename + ", "
                                 + libraw_strerror(ret) + "\r\n");
    }
    clog::L(__func__, 2) << "Image size: " << S.width << " x " << S.height
                         << ", Raw size: " << S.raw_width << " x " << S.raw_height << std::endl;
    clog::L(__func__, 2) << "Margins: top = " << S.top_margin << ", left = " << S.left_margin << std::endl;

    if ((ret = RawProcessor.unpack()) != LIBRAW_SUCCESS) {
        throw std::runtime_error(std::string("Cannot unpack file ") + filename + ", "
                                 + libraw_strerror(ret) + "\r\n");
    }

    clog::L(__func__, 2) << "Unpacked...." << std::endl;
    clog::L(__func__, 2) << "Color matrix (top left corner):" << std::endl;
    for (size_t ii = 0; ii < 6 && ii < S.width; ++ii) {
        for (size_t jj = 0; jj < 6 && jj < S.height; ++jj) {
            clog::L(__func__, 2) << RawProcessor.COLOR(ii, jj) << ":"
                                 << RawProcessor.imgdata.idata.cdesc[RawProcessor.COLOR(ii, jj)] << " ";
        }
        clog::L(__func__, 2) << std::endl;
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

    clog::L(__func__, 2) << "model2: " << RawProcessor.imgdata.color.model2 << std::endl;
    clog::L(__func__, 2) << "UniqueCameraModel: " << RawProcessor.imgdata.color.UniqueCameraModel << std::endl;
    clog::L(__func__, 2) << "LocalizedCameraModel: " << RawProcessor.imgdata.color.LocalizedCameraModel << std::endl;

    clog::L(__func__, 2) << "desc: " << RawProcessor.imgdata.other.desc << std::endl;
    clog::L(__func__, 2) << "artist: " << RawProcessor.imgdata.other.artist << std::endl;

    clog::L(__func__, 2) << "make: " << RawProcessor.imgdata.idata.make << std::endl;
    clog::L(__func__, 2) << "model: " << RawProcessor.imgdata.idata.model << std::endl;


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
            && ("ILCE-7RM2" == std::string(RawProcessor.imgdata.idata.model)
                || "DSC-RX10M3" == std::string(RawProcessor.imgdata.idata.model))) {
        clog::L(__func__, 2) << "Known camera detected, scaling result" << std::endl;
        //result *= 4;
        double min = 0, max = 0;
        cv::minMaxIdx(result, &min, &max);
        clog::L(__func__, 2) << "original min/max: " << min << " / " << max << std::endl;
        result.forEach<uint16_t>([&](uint16_t& element, const int []) -> void
        {
            element *= 3;
        }
        );
        cv::minMaxIdx(result, &min, &max);
        clog::L(__func__, 2) << "scaled min/max: " << min << " / " << max << std::endl;
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
    clog::L(__func__, 2) << "Image size: " << S.width << "x" << S.height << std::endl
                         << "Raw size: " << S.raw_width << "x" << S.raw_height << std::endl
                         << "Margins: top=" << S.top_margin << ", left=" << S.left_margin << std::endl;

    return cv::Size(S.width, S.height);
}

void Calib::paintSubmarkers(std::vector<Corner> const& submarkers, cv::Mat& image, const int paint_size_factor) const {
    size_t paint_counter = 0;
    for(size_t ii = 0; ii < submarkers.size(); ++ii) {
        Corner const& c = submarkers[ii];

        // Ignore markers outside the image (may occur in simulation)
        if (c.p.x < 0 || c.p.y < 0 || c.p.x * paint_size_factor + 1 > image.cols || c.p.y * paint_size_factor + 1 > image.rows) {
            continue;
        }
        paint_counter++;

        cv::Point2f local_shift(0,0);
        if (recursionDepth == 1 && c.id.x % 10 != 0 && c.id.y % 10 == 0) {
            if (c.id.x % 10 == 3 || c.id.x % 10 == 7) {
                local_shift.y = 16;
            }
        }

        cv::Scalar const& font_color = color_circle[ii % color_circle.size()];
        std::string const text = to_string(c.id.x) + "/" + to_string(c.id.y) + "/" + to_string(c.page);
        circle(image, paint_size_factor*c.p, 1, Scalar(0,0,0,127), 2);
        circle(image, paint_size_factor*c.p, 1, Scalar(0,255,0,127));
        putText(image, text.c_str(), local_shift+paint_size_factor*c.p, FONT_HERSHEY_PLAIN, .9, Scalar(0,0,0,0), 2, cv::LINE_AA);
        putText(image, text.c_str(), local_shift+paint_size_factor*c.p, FONT_HERSHEY_PLAIN, .9, font_color, 1, cv::LINE_AA);
        std::string const text_page = to_string(c.page);
        double font_size = 2;
        if (c.id.x % cornerIdFactor == 0 && c.id.y % cornerIdFactor == 0) {
            cv::Point2f const point_page = paint_size_factor * (c.p + cv::Point2f(c.size/2 - font_size*5, c.size/2 - font_size*5));
            putText(image, text_page.c_str(), point_page, FONT_HERSHEY_PLAIN, font_size, Scalar(0,0,0,0), 2, cv::LINE_AA);
            putText(image, text_page.c_str(), point_page, FONT_HERSHEY_PLAIN, font_size, color_circle[c.page % color_circle.size()], 1, cv::LINE_AA);
        }
    }
}

void Calib::paintSubmarkersRMS_SNR(
        const string &prefix,
        const std::vector<Corner> &submarkers,
        cv::Size const size,
        const int paint_size_factor) const {
    size_t paint_counter = 0;
    cv::Mat_<float> rms(size, float(0)), snr(size, float(0));
    cv::Mat_<uint8_t> valid_mask = cv::imread("valid-mask.png", cv::IMREAD_GRAYSCALE);
    runningstats::QuantileStats<float> valid_snr, valid_rms, invalid_snr, invalid_rms;
    for(size_t ii = 0; ii < submarkers.size(); ++ii) {
        Corner const& c = submarkers[ii];

        // Ignore markers outside the image (may occur in simulation)
        if (c.p.x < 0 || c.p.y < 0 || c.p.x * paint_size_factor + 1 > rms.cols || c.p.y * paint_size_factor + 1 > rms.rows) {
            continue;
        }

        cv::Point2i pt(std::round(c.p.x), std::round(c.p.y));

        if (c.rms > 0) {
            circle(rms, paint_size_factor*c.p, 2, Scalar(c.rms, c.rms, c.rms, c.rms), cv::FILLED, cv::LINE_AA);
            if (valid_mask.size() == size) {
                if (valid_mask(pt) > 127) {
                    valid_rms.push_unsafe(c.rms);
                }
                else {
                    invalid_rms.push_unsafe(c.rms);
                }
            }
        }
        if (c.snr > 10) {
            circle(snr, paint_size_factor*c.p, 2, Scalar(c.snr, c.snr, c.snr, c.snr)/10, cv::FILLED, cv::LINE_AA);
            if (valid_mask.size() == size) {
                if (valid_mask(pt) > 127) {
                    valid_snr.push_unsafe(c.snr);
                }
                else {
                    invalid_snr.push_unsafe(c.snr);
                }
            }
        }
    }
    clog::L(__func__, 2) << "Saving RMS and SNR images";
    cv::imwrite(prefix + "-sub-snr.tif", snr);
    cv::imwrite(prefix + "-sub-rms.tif", rms);
    if (valid_mask.size() == size) {
        runningstats::HistConfig conf;
        conf.setDataLabel("RMSE");
        conf.setMinMaxX(0, 10);
        valid_rms.plotHistAndCDF(prefix + "-rms-valid", -1, conf);
        invalid_rms.plotHistAndCDF(prefix + "-rms-invalid", -1, conf);

        conf.setDataLabel("SNR");
        conf.setMinMaxX(0, 35);
        valid_snr.plotHistAndCDF(prefix + "-snr-valid", -1, conf);
        invalid_snr.plotHistAndCDF(prefix + "-snr-invalid", -1, conf);

        /*
        conf.setLogX().setLogY();

        conf.setDataLabel("RMSE");
        conf.setMinMaxX(0, 10);
        valid_rms.plotHistAndCDF(prefix + "-log-valid-rms", -1, conf);
        invalid_rms.plotHistAndCDF(prefix + "-log-invalid-rms", -1, conf);

        conf.setDataLabel("SNR");
        conf.setMinMaxX(0, 35);
        valid_snr.plotHistAndCDF(prefix + "-log-valid-snr", -1, conf);
        invalid_snr.plotHistAndCDF(prefix + "-log-invalid-snr", -1, conf);
        */
    }
}

void Calib::initializeCameraMatrix(const double focal_length,
                                   const double cx,
                                   const double cy) {
    calibrations["OpenCV"];
    for (auto & it : calibrations) {
        it.second.cameraMatrix = cv::Mat_<double>::eye(3,3);
        it.second.cameraMatrix(0,0) = it.second.cameraMatrix(1,1) = focal_length;
        it.second.cameraMatrix(0,2) = cx;
        it.second.cameraMatrix(1,2) = cy;
    }
}

void Calib::initialzeDistortionCoefficients() {
    calibrations["OpenCV"];
    for (auto& it : calibrations) {
        it.second.distCoeffs *= 0;
    }
}

CalibResult &Calib::getCalib(const string &name) {
    auto it = calibrations.find(name);
    if (it != calibrations.end()) {
        return it->second;
    }
    throw std::runtime_error(std::string("Could not find calibration named ") + name);
}

void Calib::write(FileStorage &fs) const {

    fs << "{"
       << "imageSize" << imageSize
       << "resolutionKnown" << resolutionKnown
       << "apertureWidth" << apertureWidth
       << "apertureHeight" << apertureHeight
       << "useOnlyGreen" << useOnlyGreen
       << "recursionDepth" << recursionDepth
       << "effort" << effort
       << "demosaic" << demosaic
       << "libraw" << libraw
       << "markerSize" << markerSize;

    fs << "calibrations" << "{";
    for (auto const& it : calibrations) {
        fs << it.first << it.second;
    }
    fs << "}";

    if (!validPages.empty()) {
        fs << "validPages" << validPages;
    }

    fs << "images" << "[";
    for (const std::pair<const std::string, CornerStore>& it : data) {
        fs << "{"
           << "name" << it.first
           << "}";
        if (!fs::is_regular_file(it.first + "-submarkers-clean.hdmarker.gz")) {
            Corner::writeFile(it.first + "-submarkers-clean.hdmarker.gz", it.second.getCorners());
        }
    }
    fs << "]";

    fs << "}";
}

void Calib::read(const FileNode &node) {
    node["imageSize"] >> imageSize;
    node["resolutionKnown"] >> resolutionKnown;
    node["apertureWidth"] >> apertureWidth;
    node["apertureHeight"] >> apertureHeight;
    node["useOnlyGreen"] >> useOnlyGreen;
    node["effort"] >> effort;
    node["demosaic"] >> demosaic;
    node["libraw"] >> libraw;
    node["markerSize"] >> markerSize;
    setRecursionDepth(recursionDepth);

    FileNode n = node["validPages"]; // Read string sequence - Get node
    if (n.type() == FileNode::SEQ) {
        for (FileNodeIterator it = n.begin(); it != n.end(); ++it) {
            int val = 0;
            *it >> val;
            validPages.push_back(val);
        }
    }


    n = node["images"]; // Read string sequence - Get node
    if (n.type() != FileNode::SEQ) {
        throw std::runtime_error("Error while reading cached calibration result: Images is not a sequence. Aborting.");
    }

    std::vector<std::string> local_filenames;
    for (FileNodeIterator it = n.begin(); it != n.end(); ++it) {
        std::string name;
        (*it)["name"] >> name;
        local_filenames.push_back(name);
        addInputImage(name, std::vector<Corner>());
    }
#pragma omp parallel for
    for (size_t ii = 0; ii < local_filenames.size(); ++ii) {
        std::string const& name = local_filenames[ii];
        bool is_clean = false;
        data[name].replaceCorners(getSubMarkers(name, effort, demosaic, libraw, &is_clean));
        if (!is_clean) {
            data[name].clean(cornerIdFactor);
        }
    }
    if (local_filenames.size() != data.size()) {
        throw std::runtime_error(std::string("Error in Calib::read: data.size() = ")
                                 + std::to_string(data.size()) + " but local_filenames.size() = "
                                 + std::to_string(local_filenames.size()));
    }

    std::cout << "Number of corners removed by CornerStore::clean:" << std::endl;
    for (auto const& it : data) {
        std::cout << it.second.lastCleanDifference() << " ";
    }
    std::cout << std::endl;

    n = node["calibrations"];
    if (n.isMap()) {
        for(cv::FileNodeIterator it = n.begin(); it != n.end(); ++it) {
            cv::FileNode node = *it;
            std::string const key = node.name();
            node >> calibrations[key];
            calibrations[key].calib = this;
            calibrations[key].name = key;
        }
    }
    if (!calibrations.empty()) {
        imageFiles = calibrations.begin()->second.imageFiles;
    }
}

Mat CalibResult::getRVec(const string &filename) const {
    assert(imageFiles.size() == rvecs.size());
    for (size_t ii = 0; ii < imageFiles.size(); ++ii) {
        if (filename == imageFiles[ii]) {
            return rvecs[ii];
        }
    }
    throw std::runtime_error(std::string("CalibResult does not contain the requested file ") + filename);
}

void CalibResult::scaleResult(const double ratio) {
    for (cv::Mat& m : tvecs) {
        m *= ratio;
    }
    for (std::pair<const cv::Scalar_<int>, cv::Point3f> & it : objectPointCorrections) {
        it.second *= ratio;
    }
}

Mat CalibResult::getTVec(const string &filename) const {
    assert(imageFiles.size() == tvecs.size());
    for (size_t ii = 0; ii < imageFiles.size(); ++ii) {
        if (filename == imageFiles[ii]) {
            return tvecs[ii];
        }
    }
    throw std::runtime_error(std::string("CalibResult does not contain the requested file ") + filename);
}

std::vector<double> CalibResult::getErrorPercentiles() {
    if (error_percentiles.size() >= 101) {
        return error_percentiles;
    }
    std::vector<cv::Point2d> reprojections;
    std::vector<Corner> markers;
    getAllReprojections(markers, reprojections);
    assert(error_percentiles.size() >= 101);
    return error_percentiles;
}

void CalibResult::keepMarkers(CornerStore const& keep) {
    std::map<cv::Scalar_<int>, cv::Point3f, cmpScalar> new_objectPointCorrections;
    for (std::pair<const cv::Scalar_<int>, cv::Point3f> const& it : objectPointCorrections) {
        hdmarker::Corner c;
        if (keep.hasIDLevel(it.first, c)) {
            new_objectPointCorrections[it.first] = it.second;
        }
    }
    objectPointCorrections = new_objectPointCorrections;
}

void CalibResult::write(FileStorage &fs) const {
    fs << "{"
       << "cameraMatrix" << cameraMatrix
       << "distCoeffs" << distCoeffs
       << "distN" << distN
       << "spline_x" << spline_x
       << "spline_y" << spline_y
       << "x_factor" << x_factor
       << "error_percentiles" << error_percentiles
       << "error_median" << error_median
       << "inverseDistCoeffs" << inverseDistCoeffs;
    fs << "outlier_percentages" << outlier_percentages;
    fs << "rectification" << rectification;

    fs << "images" << "[";
    for (size_t ii = 0; ii < imageFiles.size(); ++ii) {
        fs << "{"
           << "name" << imageFiles[ii]
              << "rvec" << rvecs[ii]
                 << "tvec" << tvecs[ii]
                    << "}";
    }
    fs << "]"
       << "objectPointCorrections" << "[";
    for (const auto& it : objectPointCorrections) {
        double const sq_norm = it.second.dot(it.second);
        if (sq_norm > 1e-100) {
            fs << "{"
               << "id" << it.first
               << "val" << it.second
               << "}";
        }
    }
    fs << "]"
       << "raw_objectPointCorrections" << "[";
    for (const auto& it : raw_objectPointCorrections) {
        double const sq_norm = it.second.dot(it.second);
        if (sq_norm > 1e-100) {
            fs << "{"
               << "id" << it.first
               << "val" << it.second
               << "}";
        }
    }
    fs << "]";
    fs << "}";
}

void CalibResult::read(const FileNode &node) {
    node["cameraMatrix"] >> cameraMatrix;
    node["distCoeffs"] >> distCoeffs;
    node["distN"] >> distN;
    node["rectification"] >> rectification;
    node["outlier_percentages"] >> outlier_percentages;
    node["inverseDistCoeffs"] >> inverseDistCoeffs;
    node["x_factor"] >> x_factor;
    node["spline_x"] >> spline_x;
    node["spline_y"] >> spline_y;
    node["error_percentiles"] >> error_percentiles;
    node["error_median"] >> error_median;
    FileNode n = node["objectPointCorrections"]; // Read string sequence - Get node
    if (n.type() != FileNode::SEQ) {
        throw std::runtime_error("Error while reading cached calibration result: objectPointCorrections is not a sequence. Aborting.");
    }
    for (FileNodeIterator it = n.begin(); it != n.end(); ++it) {
        cv::Scalar_<int> id;
        cv::Point3f val;
        (*it)["id"] >> id;
        (*it)["val"] >> val;
        objectPointCorrections[id] = val;
    }
    n = node["raw_objectPointCorrections"]; // Read string sequence - Get node
    if (n.type() == FileNode::SEQ) {
        for (FileNodeIterator it = n.begin(); it != n.end(); ++it) {
            cv::Scalar_<int> id;
            cv::Point3f val;
            (*it)["id"] >> id;
            (*it)["val"] >> val;
            raw_objectPointCorrections[id] = val;
        }
    }
    n = node["images"]; // Read string sequence - Get node
    if (n.type() != FileNode::SEQ) {
        // throw std::runtime_error("Error while reading cached calibration result: Images is not a sequence. Aborting.");
    }

    for (FileNodeIterator it = n.begin(); it != n.end(); ++it) {
        cv::Mat rvec, tvec;
        std::string name;
        (*it)["name"] >> name;
        (*it)["rvec"] >> rvec;
        (*it)["tvec"] >> tvec;
        imageFiles.push_back(name);
        rvecs.push_back(rvec);
        tvecs.push_back(tvec);
        //cv::FileNode corners_node = (*it)["corners"];
    }
}


void Calib::white_balance_inplace(Mat &mat, const Point3f white) {
    float const min_val = std::min(white.x, std::min(white.y, white.z));
    //float const max_val = std::max(white.x, std::max(white.y, white.z));
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

void write(FileStorage &fs, const string &, const CalibResult &x) {
    x.write(fs);
}

void read(const FileNode &node, CalibResult &x, const CalibResult &default_value) {
    if(node.empty()) {
        throw std::runtime_error("Could not recover calibration, file storage is empty.");
    }
    else
        x.read(node);
}

#if 0 // This is disfunctional anyway
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
#endif

void Calib::refineRecursiveByPage(Mat &img,
                                  const std::vector<Corner> &in,
                                  std::vector<Corner> &out,
                                  int const recursion_depth,
                                  double &markerSize,
                                  bool use_three) {
    std::map<int, std::vector<hdmarker::Corner> > pages;
    for (const hdmarker::Corner& c : in) {
        pages[c.page].push_back(c);
    }
    out.clear();
    double _markerSize = markerSize;
    clog::L(__func__, 2) << "Input page distribution: " << std::endl;
    printCornerStats(in);
    hdmarker::Marker::init();
    cv::Mat_<float> clone(img.clone());
    for (const auto& it : pages) {
        _markerSize = markerSize;
        std::vector<hdmarker::Corner> _out, _rejected;
        std::vector<cv::Rect> limits = {{0, 0, 31, 31}};
        if (use_three) {
            hdmarker::refine_recursive_3(clone, in, _out, _rejected, recursion_depth, &_markerSize, nullptr, nullptr, it.first, limits);
        }
        else {
            hdmarker::refine_recursive(clone, in, _out, _rejected, recursion_depth, &_markerSize, nullptr, nullptr, it.first, limits);
        }
        out.insert(out.end(), _out.begin(), _out.end());
        clog::L(__func__, 2) << "Refining page " << it.first
                             << ", recursion depth " << recursion_depth
                             << " got " << _out.size() << " new submarkers, current total: "
                             << out.size();
        printCornerStats(_out);
    }
    markerSize = _markerSize;
}

} // namespace hdcalib
