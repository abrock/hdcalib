#include <iostream>

#include <exception>
#include <boost/filesystem.hpp>

#include <tclap/CmdLine.h>
#include <ParallelTime/paralleltime.h>

#include "hdcalib.h"

namespace fs = boost::filesystem;

void trim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
        return !std::isspace(ch);
    }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

boost::system::error_code error_code;

int main(int argc, char ** argv) {

    clog::Logger::getInstance().addListener(std::cout);

    hdcalib::Calib calib;
    std::map<std::string, std::vector<std::string> > input_files;
    bool demosaic = false;
    bool libraw = false;
    bool only_green = false;
    fs::path output_dir;
    std::string cache_file;
    std::string calibName = "Flexible";

#if CATCH_EXCEPTIONS
    try {
#endif
        TCLAP::CmdLine cmd("hdcalib calibration tool", ' ', "0.1");

        TCLAP::ValueArg<std::string> cache_arg("c", "cache",
                                               "Cache file for the calibration results. This makes use of the opencv filestorage capabilities so filename extension should be .xml/.xml.gz/.yaml/.yaml.gz",
                                               true, "", "Calibration cache.");
        cmd.add(cache_arg);

        TCLAP::ValueArg<std::string> output_arg("o", "output",
                                               "Output directory for rectified images. Subdirectories will be created automatically as needed.",
                                               true, "", "Output directorys.");
        cmd.add(output_arg);

        TCLAP::ValueArg<std::string> calibNameArg("n", "name",
                                               "Name of the calibration result to use. Options include OpenCV, Ceres, Flexible and SemiFlexible",
                                               false, "Flexible", "");
        cmd.add(calibNameArg);

        TCLAP::SwitchArg demosaic_arg("d", "demosaic",
                                      "Use this flag if the input images are raw images and demosaicing should be used.",
                                      false);
        cmd.add(demosaic_arg);

        TCLAP::SwitchArg read_raw_arg("", "raw",
                                      "Use this flag if the input images are raw images which must be read using LibRaw since OpenCV cannot read them. This implies -d.",
                                      false);
        cmd.add(read_raw_arg);

        TCLAP::SwitchArg only_green_arg("g", "only-green", "Set this flag true if only the green channel of a bayer image should be used."
                                                           "In the case of demosaicing this means that the missing green pixels"
                                                           "are interpolated bilinear.", false);
        cmd.add(only_green_arg);

        TCLAP::MultiArg<std::string> textfile_arg("i",
                                                  "input",
                                                  "Text file containing a list of image paths relative to the working directory.",
                                                  false,
                                                  "Text file with a list of input images.");
        cmd.add(textfile_arg);

        TCLAP::UnlabeledMultiArg<std::string> input_img_arg("input_img", "Input images, should contain markers.", false, "Input images.");
        cmd.add(input_img_arg);

        cmd.parse(argc, argv);


        input_files[""] = input_img_arg.getValue();

        libraw = read_raw_arg.getValue();
        only_green = only_green_arg.getValue();
        demosaic = demosaic_arg.getValue() || libraw;
        std::vector<std::string> const textfiles = textfile_arg.getValue();
        cache_file = cache_arg.getValue();


        output_dir = output_arg.getValue();
        fs::create_directories(output_dir, error_code);

        for (std::string const& file : textfiles) {
            if (!fs::is_regular_file(file)) {
                continue;
            }
            std::ifstream in(file);
            std::string line;
            while (std::getline(in, line)) {
                trim(line);
                if (fs::is_regular_file(line)) {
                    input_files[file].push_back(line);
                    fs::path p(line);
                    if (p.has_parent_path()) {
                        fs::create_directories(output_dir / p.parent_path());
                    }
                }
            }
        }

        if (input_files.empty()) {
            std::cerr << "Fatal error: No input files specified." << std::endl;
            cmd.getOutput()->usage(cmd);

            return EXIT_FAILURE;
        }

        calibName = calibNameArg.getValue();
        std::cout << "Parameters: " << std::endl
                  << "Number of input files: " << input_files.size() << std::endl
                  << "demosaic: " << (demosaic ? "true" : "false") << std::endl
                  << "use libraw: " << (libraw ? "true" : "false") << std::endl
                  << "only green channel: " << (only_green ? "true" : "false") << std::endl
                  << "output directory: " << output_dir << std::endl
                  << "cache file: " << cache_file << std::endl
                  << "calib name: " << calibName << std::endl;

        calib.only_green(only_green);
#if CATCH_EXCEPTIONS
    }
    catch (TCLAP::ArgException const & e) {
        std::cerr << e.what() << std::endl;
        return 0;
    }
    catch (...) {
        std::cerr << "Unknown exception" << std::endl;
        return 0;
    }
#endif

    bool has_cached_calib = false;
    if (fs::is_regular_file(cache_file)) {
        try {
            clog::L(__func__, 2) << "Reading cached calibration results..." << std::flush;
            cv::FileStorage fs(cache_file, cv::FileStorage::READ);
            cv::FileNode n = fs["calibration"];
            n >> calib;
            has_cached_calib = true;
            fs.release();
            std::cout << " done." << std::endl;
            calib.purgeInvalidPages();
        }
        catch (std::exception const& e) {
            std::cout << "Reading cache file failed with exception:" << std::endl
                      << e.what() << std::endl;
            return EXIT_FAILURE;
        }
        catch (...) {
            std::cout << "Unknown exception." << std::endl;
            return EXIT_FAILURE;
        }
    }
    else {
        std::cout << "Calibration cache argument must be a regular file." << std::endl;
        return EXIT_FAILURE;
    }

    const cv::Mat_<cv::Vec2f> remap = calib.getCachedUndistortRectifyMap(calibName);

    std::vector<std::string> images;
    for (const auto& it : input_files) {
        for (const auto& file : it.second) {
            images.push_back(file);
        }
    }

    size_t counter = 0;
#pragma omp parallel for schedule(dynamic)
    for (size_t ii = 0; ii < images.size(); ++ii) {
        std::string const& file = images[ii];
#pragma omp critical
        {
            clog::L(__func__, 1) << "Remapping file " << ++counter << " out of " << images.size() << ": " << file << std::endl;
        }
        const cv::Mat img = hdcalib::Calib::readImage(file, demosaic, libraw, only_green);
        cv::Mat remapped;
        cv::remap(img, remapped, remap, cv::Mat(), cv::INTER_LINEAR);
        fs::path output_path = output_dir / (file + "-rect.png");
        cv::imwrite(output_path.string(), remapped);
    }

    return EXIT_SUCCESS;
}
