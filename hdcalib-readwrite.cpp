#include "hdcalib.h"

#include <runningstats/runningstats.h>

namespace hdcalib {

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

void Calib::addInputImage(const string filename, const std::vector<Corner> &corners) {
    invalidateCache();
    CornerStore & ref = data[filename];
    ref.replaceCorners(corners);
    ref.clean(cornerIdFactor);
}

void Calib::addInputImageAfterwards(const string filename, const std::vector<Corner> &corners) {
    if (hasFile(filename)) {
        if (verbose) {
            std::cout << "Not adding duplicate image " << filename << std::endl;
        }
        return;
    }
    if (verbose) {
        std::cout << "Adding image " << filename << "..." << std::flush;
    }
    invalidateCache();

    rvecs.push_back(cv::Mat());
    tvecs.push_back(cv::Mat());
    imageFiles.push_back(filename);

    insertSorted(imageFiles, rvecs, tvecs);

    std::cout << "replacing corners..." << std::flush;
    CornerStore & ref = data[filename];
    ref.replaceCorners(validPages.empty() ? corners : purgeInvalidPages(corners, validPages));
    ref.clean(cornerIdFactor);

    size_t index = 0;
    for (; index < imageFiles.size(); ++index) {
        if (filename == imageFiles[index]) {
            break;
        }
    }

    std::cout << "preparing calibration..." << std::flush;
    prepareCalibration();
    std::cout << "solvePnP..." << std::flush;
    bool const success = cv::solvePnP (
                objectPoints[index],
                imagePoints[index],
                cameraMatrix,
                distCoeffs,
                rvecs[index],
                tvecs[index]);
    std::cout << "done." << std::endl;
}

void Calib::addInputImage(const string filename, const std::vector<Corner> &corners, cv::Mat const& rvec, cv::Mat const& tvec) {
    if (hasFile(filename)) {
        return;
    }
    invalidateCache();

    CornerStore & ref = data[filename];
    ref.replaceCorners(validPages.empty() ? corners : purgeInvalidPages(corners, validPages));
    ref.clean(cornerIdFactor);
    rvecs.push_back(rvec);
    tvecs.push_back(tvec);
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

void Calib::initializeCameraMatrix(const double focal_length, const double cx, const double cy) {
    cameraMatrix = cv::Mat_<double>::eye(3,3);
    cameraMatrix(0,0) = cameraMatrix(1,1) = focal_length;
    cameraMatrix(0,2) = cx;
    cameraMatrix(1,2) = cy;
}

void Calib::initialzeDistortionCoefficients() {
    distCoeffs *= 0;
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

    if (!validPages.empty()) {
        fs << "validPages" << "[";
        for (auto const val : validPages) {
            fs << val;
        }
        fs << "]";
    }

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
    node["recursionDepth"] >> recursionDepth;
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
        const FileNodeIterator corner_end = corners_node.end();
        for (FileNodeIterator corner_it = corners_node.begin(); corner_it != corner_end; ++corner_it) { // Go through the node
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

} // namespace hdcalib
