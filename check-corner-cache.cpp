#include <iostream>

#include <hdmarker/hdmarker.hpp>

#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

void checkCache(std::string const filename, std::string const type) {
    std::vector<hdmarker::Corner> hdm_corners, cv_corners;
    std::string const hdm_file = filename + type + "hdmarker.gz";
    std::string const cv_file = filename + type + "yaml.gz";
    if (fs::is_regular_file(hdm_file)) {
        try {
            hdmarker::Corner::readFile(hdm_file, hdm_corners);
        }  catch (...) {}
    }
    if (fs::is_regular_file(cv_file)) {
        try {
            cv::FileStorage storage(cv_file, cv::FileStorage::READ);
            storage["corners"] >> cv_corners;
        }  catch (...) {}
    }
    if (cv_corners.size() > hdm_corners.size()) {
        hdmarker::Corner::writeGzipFile(hdm_file, cv_corners);
        std::cout << "Copying CV to HDM for file " << filename << ", sizes are " << cv_corners.size() << " / " << hdm_corners.size() << std::endl;
    }
    else if (cv_corners.size() < hdm_corners.size()) {
        cv::FileStorage storage(cv_file, cv::FileStorage::WRITE);
        storage << "corners" << hdm_corners;
        std::cout << "Copying HDM to CV for file " << filename << ", sizes are " << hdm_corners.size() << " / " << cv_corners.size() << std::endl;
    }
    else {
        std::cout << "No discrepancy for file " << filename << ", sizes are " << hdm_corners.size() << " / " << cv_corners.size() << std::endl;
    }
}

int main(int argc, char ** argv) {

    for (size_t ii = 1; ii < size_t(argc); ++ii) {
        checkCache(argv[ii], "-pointcache.");
        checkCache(argv[ii], "-submarkers.");
    }
    std::string line;
    while (std::getline(std::cin, line)) {

    }
}
