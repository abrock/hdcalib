#include "griddescription.h"

namespace hdcalib {

GridPointDesc::GridPointDesc(const std::string _suffix, const cv::Vec3d _point) : suffix(_suffix), point(_point) {}

GridPointDesc::GridPointDesc() {}

cv::Point3f GridPointDesc::getPt() const {
    return cv::Point3f(point[0], point[1], point[2]);
}

void GridPointDesc::write(cv::FileStorage &fs) const {
    fs << "{"
       << "suffix" << suffix
       << "point" << point
       << "}";
}

void GridPointDesc::read(const cv::FileNode &node) {
    node["suffix"] >> suffix;
    node["point"] >> point;
}

GridPointDesc GridDescription::getDesc(const std::string &suffix) const {
    for (size_t ii = 0; ii < points.size(); ++ii) {
        if (suffix == points[ii].suffix) {
            return points[ii];
        }
    }
    throw std::runtime_error(std::string("Couldn't find gridpoint with suffix ") + suffix);
}

void GridDescription::write(cv::FileStorage &fs) const {
    fs << "{"
       << "name" << name
       << "points" << points
       << "fixed_scale" << fixed_scale
       << "}";
}

void GridDescription::read(const cv::FileNode &node) {
    node["name"] >> name;
    node["points"] >> points;
    node["fixed_scale"] >> fixed_scale;
}

void GridDescription::readFile(const std::string filename, std::vector<GridDescription> &data) {
    cv::FileStorage storage(filename, cv::FileStorage::READ);
    std::vector<GridDescription> new_data;
    storage["grids"] >> new_data;
    if (new_data.empty()) {
        throw std::runtime_error(std::string("File ") + filename + " doesn't contain any readable grid descriptions");
    }
    data.insert(data.end(), new_data.begin(), new_data.end());
}

void GridDescription::writeFile(const std::string filename, const std::vector<GridDescription> &data) {
    cv::FileStorage storage(filename, cv::FileStorage::WRITE);
    storage << "grids" << data;
}

void write(cv::FileStorage &fs, const std::string &, const hdcalib::GridPointDesc &x) {
    x.write(fs);
}

void read(const cv::FileNode &node, hdcalib::GridPointDesc &x, const hdcalib::GridPointDesc &default_value) {
    if(node.empty()) {
        x = default_value;
    }
    else
        x.read(node);
}

void write(cv::FileStorage &fs, const std::string &, const GridDescription &x) {
    x.write(fs);
}

void read(const cv::FileNode &node, GridDescription &x, const GridDescription &default_value) {
    if(node.empty()) {
        x = default_value;
    }
    else
        x.read(node);
}

} // namespace hdcalib
