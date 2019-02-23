#include "calib.h"


namespace hdcalib {


Calib::Calib()
{

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
            if (ii % 2 == 0) { // "blue" row (we dont actually care if it's blue or red)
                if (jj % 2 == 0) {
                    max_g = std::max(max_g, row[jj]);
                }
                else {
                    max_b = std::max(max_b, row[jj]);
                }
            }
            else { // Red row
                if (jj % 2 == 0) {
                    max_r = std::max(max_r, row[jj]);
                }
                else {
                    max_g = std::max(max_g, row[jj]);
                }
            }
        }
    }
    const double f_r = 65535.0/max_r;
    const double f_g = 65535.0/max_g;
    const double f_b = 65535.0/max_b;

    for (int ii = 0; ii < input.rows; ++ii) {
        unsigned short * row = input.ptr<unsigned short>(ii);
        for (int jj = 0; jj < input.cols; ++jj) {
            if (ii % 2 == 0) { // "blue" row (we dont actually care if it's blue or red)
                if (jj % 2 == 0) {
                    row[jj] = cv::saturate_cast<unsigned short>(std::round(row[jj] * f_g));
                }
                else {
                    row[jj] = cv::saturate_cast<unsigned short>(std::round(row[jj] * f_b));
                }
            }
            else { // Red row
                if (jj % 2 == 0) {
                    row[jj] = cv::saturate_cast<unsigned short>(std::round(row[jj] * f_r));
                }
                else {
                    row[jj] = cv::saturate_cast<unsigned short>(std::round(row[jj] * f_g));
                }
            }
        }
    }

}

Mat Calib::read_raw(const string &filename) {
    LibRaw RawProcessor;


    auto& S = RawProcessor.imgdata.sizes;
    auto& OUT = RawProcessor.imgdata.params;

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

    if ((ret = RawProcessor.unpack()) != LIBRAW_SUCCESS) {
        throw std::runtime_error(std::string("Cannot unpack file ") + filename + ", "
                                 + libraw_strerror(ret) + "\r\n");
    }

    if (verbose)
        printf("Unpacked....\n");

    if (!(RawProcessor.imgdata.idata.filters || RawProcessor.imgdata.idata.colors == 1)) {
        throw std::runtime_error(
                    std::string("Only Bayer-pattern RAW files supported, file ") + filename
                    + " seems to have a different pattern.\n");
    }

    cv::Mat result(S.raw_height, S.raw_width, CV_16UC1);

    for (int jj = 0, global_counter = 0; jj < S.raw_height ; jj++) {
        unsigned short * row = result.ptr<unsigned short>(jj);
        for (int ii = 0; ii < S.raw_width; ++ii, ++global_counter) {
            row[ii] = RawProcessor.imgdata.rawdata.raw_image[global_counter];
        }
    }


    return result;
}

vector<Corner> Calib::getCorners(const std::string input_file,
        const float effort,
        const bool demosaic,
        const int recursion_depth,
        const bool raw) {
    std::string pointcache_file = input_file + "-pointcache.yaml";
    vector<Corner> corners;
    bool read_cache_success = false;



    try {
        if (fs::exists(pointcache_file)) {
            read_cache_success = true;
            FileStorage pointcache(pointcache_file, FileStorage::READ);
            FileNode n = pointcache["corners"]; // Read string sequence - Get node
            if (n.type() != FileNode::SEQ)
            {
                cerr << "corners is not a sequence! FAIL" << endl;
                read_cache_success = false;
            }

            FileNodeIterator it = n.begin(), it_end = n.end(); // Go through the node
            for (; it != it_end; ++it) {
                hdmarker::Corner c;
                *it >> c;
                corners.push_back(c);
            }

        }
    }
    catch (const Exception& e) {
        std::cout << "Reading pointcache file failed with exception: " << std::endl
                  << e.what() << std::endl;
        read_cache_success = false;
    }

    if (read_cache_success) {
        std::cout << "Got corners from pointcache file" << std::endl;
        return corners;
    }




    //  microbench_init();

    Mat img, paint;


    if (demosaic) {
        if (raw) {
            img = read_raw(input_file);
        }
        else {
            img = cv::imread(input_file, CV_LOAD_IMAGE_GRAYSCALE);
        }
        normalize_raw_per_channel_inplace(img);
        double min_val = 0, max_val = 0;
        cv::minMaxIdx(img, &min_val, &max_val);
        std::cout << "Image min/max: " << min_val << " / " << max_val << std::endl;
        img = img * (255.0 / max_val);
        img.convertTo(img, CV_8UC1);
        //cv::normalize(img, img, 0, 255, NORM_MINMAX, CV_8UC1);
        cvtColor(img, img, COLOR_BayerBG2BGR); // RG BG GB GR
        cv::imwrite(input_file + "-demosaiced-normalized.png", img);
        paint = img.clone();
    }
    else {
        img = cv::imread(input_file);
        paint = img.clone();
    }
    std::cout << "Input image size: " << img.size << std::endl;

    Marker::init();

    detect(img, corners,use_rgb,0,10, effort, 3);

    printf("final score %zu corners\n", corners.size());

    for(Corner const& c : corners) {
        Point2f p1, p2;

        //if (c.page != 256) continue;
        char buf[256];
        sprintf(buf, "%d/%d", c.id.x, c.id.y);
        circle(paint, c.p, 1, Scalar(0,0,0,0), 2);
        circle(paint, c.p, 1, Scalar(0,255,0,0));
        putText(paint, buf, c.p, FONT_HERSHEY_PLAIN, 0.5, Scalar(0,0,0,0), 2, CV_AA);
        putText(paint, buf, c.p, FONT_HERSHEY_PLAIN, 0.5, Scalar(255,255,255,0), 1, CV_AA);
    }
    imwrite(input_file + "-1.png", paint);

    //
    Mat gray;
    if (img.channels() != 1)
        cvtColor(img, gray, CV_BGR2GRAY);
    else
        gray = img;

    if (recursion_depth > 0) {
        std::cout << "Drawing sub-markers" << std::endl;
        vector<Corner> corners_sub;
        double msize = 1.0;
        hdmarker::refine_recursive(gray, corners, corners_sub, 3, &msize);

        vector<Corner> corners_f2;

        cv::Mat paint2 = paint.clone();
        for (hdmarker::Corner const& c : corners_f2) {
            circle(paint2, c.p, 3, Scalar(0,0,255,0), -1, LINE_AA);
        }
        imwrite(input_file + "-2.png", paint2);
    }

    FileStorage pointcache(pointcache_file, FileStorage::WRITE);
    pointcache << "corners" << "[";
    for (hdmarker::Corner const& c : corners) {
        pointcache << c;
    }
    pointcache << "]";

    return corners;

}

}
