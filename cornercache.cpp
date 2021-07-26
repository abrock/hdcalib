#include "cornercache.h"

namespace hdcalib {

CornerCache &CornerCache::getInstance() {
    static CornerCache instance;
    return instance;
}

std::vector<hdmarker::Corner> &CornerCache::operator[](const std::string &filename) {
    {
        std::lock_guard<std::mutex> guard(access_mutex);
        std::lock_guard<std::mutex> guard2(access_mutex_by_file[filename]);
        std::map<std::string, std::vector<hdm::Corner> >::iterator it = data.find(filename);
        if (it != data.end()) {
            return it->second;
        }
        data[filename];
    }
    std::lock_guard<std::mutex> guard(access_mutex_by_file[filename]);
    hdcalib::Calib c;
    c.setMinSNR(snr_sigma_min);
    ParallelTime t;
    std::vector<hdm::Corner> _corners = c.getSubMarkers(filename);
    std::vector<hdm::Corner> corners;
    corners.reserve(_corners.size());
    for (hdm::Corner const& c : _corners) {
        if (std::abs(c.snr * c.getSigma()) > snr_sigma_min) {
            corners.push_back(c);
        }
    }
    //std::cout << "Reading " << filename << ": " << t.print() << std::endl;
    t.start();
    data[filename] = corners;
    //std::cout << "Creating CornerStore for " << filename << ": " << t.print() << std::endl;
    return data[filename];
}

std::vector<hdmarker::Corner> CornerCache::getGZ(const std::string &filename) {
    {
        std::lock_guard<std::mutex> guard(access_mutex);
        std::lock_guard<std::mutex> guard2(access_mutex_by_file[filename]);
        std::map<std::string, std::vector<hdm::Corner> >::iterator it = data.find(filename);
        if (it != data.end()) {
            return it->second;
        }
        data[filename];
    }
    std::lock_guard<std::mutex> guard(access_mutex_by_file[filename]);
    hdcalib::Calib c;
    c.setMinSNR(snr_sigma_min);
    ParallelTime t;
    std::vector<hdm::Corner> _corners;
    hdmarker::Corner::readGzipFile(filename, _corners);
    std::vector<hdm::Corner> corners;
    corners.reserve(_corners.size());
    for (hdm::Corner const& c : _corners) {
        if (std::abs(c.snr * c.getSigma()) > snr_sigma_min) {
            corners.push_back(c);
        }
    }
    //std::cout << "Reading " << filename << ": " << t.print() << std::endl;
    t.start();
    data[filename] = corners;
    //std::cout << "Creating CornerStore for " << filename << ": " << t.print() << std::endl;
    return data[filename];
}

void CornerCache::setSNRSigmaMin(const double val) {
    getInstance().snr_sigma_min = val;
}

}
