#ifndef HDCALIB_H
#define HDCALIB_H

#include <stdio.h>
#include <map>
#include <iostream>
#include <unordered_map>
#include <exception>

#include <hdmarker/hdmarker.hpp>
#include <hdmarker/subpattern.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <boost/filesystem.hpp>

#include <libraw/libraw.h>

#include "nanoflann.hpp"

namespace hdcalib {
using namespace std;
using namespace hdmarker;
using namespace cv;
namespace fs = boost::filesystem;

/**
 * @brief remove_duplicate_markers purges duplicate markers from a vector of markers.
 * These occur when a target has a with and/or height of 33 or more since different
 * "pages" are used on those larger targets and markers are identified as belonging
 * to both pages at the edges.
 *
 * @param in
 */
std::vector<hdmarker::Corner> filter_duplicate_markers(std::vector<hdmarker::Corner> const& in);

class CornerStore;

class CornerIndexAdaptor {
    CornerStore const& store;
public:

    CornerIndexAdaptor(CornerStore const& ref);

    /**
     * @brief kdtree_get_point_count returns corners.size()
     * @return corners.size()
     */
    size_t kdtree_get_point_count() const;

    /**
     * @brief kdtree_get_pt returns the dim'th component of the corner id.
     * The first two components are the x and y value of the id property, the third component is the page.
     *
     * @param idx index of the corner in the corner storage vector.
     * @param dim number of the dimension [0-2]
     * @return value of the requested component.
     */
    int kdtree_get_pt(const size_t idx, int dim) const;

    template <class BBOX>
    /**
     * @brief kdtree_get_bbox could optionally return a pre-computed bounding box, but at the moment no such bounding box is computed so it just returns false.
     * @param[out] bb bounding box.
     * @return false
     */
    bool kdtree_get_bbox(BBOX &bb) const {
        return false;
        /*
        bb[0].low = 0; bb[0].high = 32;  // 0th dimension limits
        bb[1].low = 0; bb[1].high = 32;  // 1st dimension limits
        bb[2].low = 0; bb[2].high = 512;  // 1st dimension limits
        return true;
        // */
    }
};

class CornerPositionAdaptor {
    CornerStore const& store;

public:
    CornerPositionAdaptor(CornerStore const& ref);

    /**
     * @brief kdtree_get_point_count returns corners.size()
     * @return corners.size()
     */
    size_t kdtree_get_point_count() const;

    /**
     * @brief kdtree_get_pt returns the dim'th component of the corner position in the image.
     * The two components are the x and y value of the "p" property of the hdmarker::Corner.
     *
     * @param idx index of the corner in the corner storage vector.
     * @param dim number of the dimension [0-1]
     * @return value of the requested component.
     */
    int kdtree_get_pt(const size_t idx, int dim) const;

    template <class BBOX>
    /**
     * @brief kdtree_get_bbox could optionally return a pre-computed bounding box, but at the moment no such bounding box is computed so it just returns false.
     * @param[out] bb bounding box.
     * @return false
     */
    bool kdtree_get_bbox(BBOX &bb) const {
        return false;
    }
};

class CornerStore {
private:
    /**
     * @brief corners contains the hdmarker::Corner objects.
     * It is important that this member is initialized first since
     * the initialization of other members depend on it.
     */
    std::vector<hdmarker::Corner> corners;

    CornerIndexAdaptor idx_adapt;
    CornerPositionAdaptor pos_adapt;

    typedef nanoflann::KDTreeSingleIndexDynamicAdaptor<
    nanoflann::L2_Simple_Adaptor<int, CornerIndexAdaptor > ,
    CornerIndexAdaptor,
    3 /* dim */
    > CornerIndexTree;

    CornerIndexTree idx_tree;

    typedef nanoflann::KDTreeSingleIndexDynamicAdaptor<
    nanoflann::L2_Simple_Adaptor<int, CornerPositionAdaptor > ,
    CornerPositionAdaptor,
    3 /* dim */
    > CornerPositionTree;

    CornerPositionTree pos_tree;

public:
    CornerStore();


    size_t size() const;

    const hdmarker::Corner & get(size_t index) const;

    /**
     * @brief push_back adds a single hdcalib::Corner to the stored corners.
     * @param x
     */
    void push_back(hdmarker::Corner const x);

    /**
     * @brief add adds a vector of hdcalib::Corner to the stored corners.
     * @param vec
     */
    void add(std::vector<hdcalib::Corner> const& vec);


};

class Calib
{
    bool verbose = true;
    std::vector<std::string> input_images;
    int grid_width = 1;
    int grid_height = 1;

    bool use_rgb = false;
public:
    Calib();

    void setInputImages(std::vector<std::string> const& files) {

    }

    cv::Mat normalize_raw_per_channel(cv::Mat const& input);

    void normalize_raw_per_channel_inplace(cv::Mat & input);

    cv::Mat read_raw(std::string const& filename);

    /**
   * @brief getCorners
   * @param input_file
   * @param effort
   * @param demosaic
   * @param recursion_depth
   * @param raw
   * @return
   */
    vector<Corner> getCorners(
            const std::string input_file,
            const float effort,
            const bool demosaic,
            const int recursion_depth,
            const bool raw);
};

}

#endif // HDCALIB_H
