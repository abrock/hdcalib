#ifndef GRIDDESCRIPTION_H
#define GRIDDESCRIPTION_H

#include <string>
#include <opencv2/core.hpp>

namespace hdcalib {

/**
 * @brief The GridPointDesc struct describes a type of point in a 3D grid by giving 3D coordinates and a filename suffix.
 */
struct GridPointDesc {
    std::string suffix;
    cv::Vec3d point;

    GridPointDesc(std::string const _suffix, cv::Vec3d const _point);
    GridPointDesc();

    /**
     * @brief write Function needed for serializating a GridPoint using the OpenCV FileStorage system.
     * @param fs
     */
    void write(cv::FileStorage& fs) const;

    /**
     * @brief read Method needed for reading a serialized GridPoint using the OpenCV FileStorage system.
     * @param node
     */
    void read(const cv::FileNode& node);
};

void write(cv::FileStorage &fs, const std::string &, const GridPointDesc &x);

void read(const cv::FileNode &node, GridPointDesc &x, const GridPointDesc &default_value = GridPointDesc());

struct GridDescription {
    std::string name;
    std::vector<GridPointDesc> points;

    GridPointDesc getDesc(std::string const& suffix) const;

    /**
     * @brief write Function needed for serializating a GridPoint using the OpenCV FileStorage system.
     * @param fs
     */
    void write(cv::FileStorage& fs) const;

    /**
     * @brief read Method needed for reading a serialized GridPoint using the OpenCV FileStorage system.
     * @param node
     */
    void read(const cv::FileNode& node);

    static void readFile(std::string const filename, std::vector<GridDescription> & data);

    static void writeFile(std::string const filename, std::vector<GridDescription> const & data);
};

void write(cv::FileStorage &fs, const std::string &, const hdcalib::GridDescription &x);

void read(const cv::FileNode &node, hdcalib::GridDescription &x, const hdcalib::GridDescription &default_value);

} // namespace hdcalib

#endif // GRIDDESCRIPTION_H
