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
    CornerStore const * store;
public:

    CornerIndexAdaptor(CornerStore &ref);

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
    CornerStore * store;

public:
    CornerPositionAdaptor(CornerStore & ref);

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
    nanoflann::L2_Simple_Adaptor<double, CornerIndexAdaptor > ,
    CornerIndexAdaptor,
    3 /* dim */
    > CornerIndexTree;

    std::shared_ptr<CornerIndexTree> idx_tree;

    typedef nanoflann::KDTreeSingleIndexDynamicAdaptor<
    nanoflann::L2_Simple_Adaptor<double, CornerPositionAdaptor > ,
    CornerPositionAdaptor,
    2 /* dim */
    > CornerPositionTree;

    std::shared_ptr<CornerPositionTree> pos_tree;

public:
    CornerStore();

    /**
     * @brief CornerStore explicit copy constructor
     * @param c
     */
    CornerStore(const CornerStore& c);

    /**
     * @brief operator = a better copy operator for avoiding memory problems.
     * @param other
     * @return
     */
    CornerStore& operator=(const CornerStore& other);

    /**
     * @brief clean first calls purgeDuplicates and then purgeUnlikely;
     */
    void clean();

    /**
     * @brief intersect calculates the intersection between this store and another store
     * and replaces this store's corners by the intersection.
     * @param b
     */
    void intersect(CornerStore const& b);

    void replaceCorners(std::vector<hdmarker::Corner> const& _corners);

    /**
     * @brief getCorners returns a copy of the stored corners.
     * @return
     */
    std::vector<hdmarker::Corner> getCorners() const;

    /**
     * @brief findByID does a nearest-neighbour search for the num_results closest hdmarker::Corner to a given marker. Distance is the L2 distance of the triples given by (id.x, id.y, page) of the corners.
     * @param ref hdmarker::Corner we are searching.
     * @param num_results maximum number of results we want. The result set might be smaller.
     * @return A vector of results, ordered by distance to ref ascending.
     */
    std::vector<hdmarker::Corner> findByID(hdmarker::Corner const& ref, size_t const num_results = 1);

    /**
     * @brief findByPos does a nearest-neighbour search for the num_results closest hdmarker::Corner to a given marker. Distance is the L2 distance of the pixel positions.
     * @param ref hdmarker::Corner we are searching.
     * @param num_results maximum number of results we want. The result set might be smaller.
     * @return A vector of results, ordered by distance to ref ascending.
     */
    std::vector<hdmarker::Corner> findByPos(hdmarker::Corner const& ref, size_t const num_results = 1);

    /**
     * @brief findByPos does a nearest-neighbour search for the num_results closest hdmarker::Corner to a given marker. Distance is the L2 distance of the pixel positions.
     * @param x-position we are searching.
     * @param y-position we are searching.
     * @param num_results maximum number of results we want. The result set might be smaller.
     * @return A vector of results, ordered by distance to ref ascending.
     */
    std::vector<hdmarker::Corner> findByPos(double const x, double const y, size_t const num_results = 1);

    /**
     * @brief purgeUnlikely searches for likely mis-detections and removes them from the store.
     */
    bool purgeUnlikely();

    /**
     * @brief purgeDuplicates removes duplicate markers.
     *
     * @return true if duplicates were found.
     */
    bool purgeDuplicates();

    /**
     * @brief hasID checks if a given hdmarker::Corner (identified by id and page) exists in the CornerStore.
     * @param ref corner we search.
     * @return true if the corner exists.
     */
    bool hasID(hdmarker::Corner const& ref) const;

    /**
     * @brief size returns the number of elements currently stored.
     * @return
     */
    size_t size() const;

    /**
     * @brief get returns a const reference a stored corner.
     * @param index
     * @return
     */
    const hdmarker::Corner & get(size_t index) const;

    /**
     * @brief push_back adds a single hdcalib::Corner to the stored corners.
     * @param x
     */
    void push_back(hdmarker::Corner const x);

    /**
     * @brief push_conditional adds the corner to the data if not a Corner with the same id is already present.
     * @param x
     */
    void push_conditional(hdmarker::Corner const x);

    /**
     * @brief add adds a vector of hdcalib::Corner to the stored corners.
     * @param vec
     */
    void add(std::vector<hdcalib::Corner> const& vec);


};

class Calib
{
    bool verbose = true;
    int grid_width = 1;
    int grid_height = 1;

    bool use_rgb = false;

    typedef std::map<std::string, CornerStore> Store_T;
    Store_T data;
public:
    Calib();

    cv::Point3f getInitial3DCoord(hdmarker::Corner const& c, double const z = 0);

    /**
     * @brief keepCommonCorners Removes all corners from the data which are not present
     * in all detected images. This is needed for the OpenCV calibration which assumes
     * checkerboard-like targets where the whole target is visible in each image.
     */
    void keepCommonCorners_delete();

    void keepCommonCorners_intersect();

    void keepCommonCorners();

    void addInputImage(std::string const filename, std::vector<hdmarker::Corner> const& corners);

    void addInputImage(std::string const filename, CornerStore const& corners);

    CornerStore get(std::string const filename) const;

    cv::Mat normalize_raw_per_channel(cv::Mat const& input);

    void normalize_raw_per_channel_inplace(cv::Mat & input);

    cv::Mat read_raw(std::string const& filename);

    CornerStore getUnion() const;

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
