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

#include <ceres/ceres.h>

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

    static void intersect(CornerStore & a, CornerStore & b);

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
     * @brief purge32 removes markers where the id's x or y component is 32 and where a second
     * marker is present at the same location.
     * @return
     */
    bool purge32();


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

    void getPoints(std::vector<cv::Point2f>& imagePoints, std::vector<cv::Point3f> & objectPoints) const;

};

class CalibrationResult {
public:
    /**
     * @brief imageFiles vector of image filenames.
     */
    std::vector<std::string> imageFiles;

    /**
     * @brief imagePoints storage for image points. The outer vector contains one inner vector per image file.
     * The inner vectors contain one cv::Point2f for each detected hdmarker::Corner.
     */
    std::vector<std::vector<cv::Point2f> > imagePoints;

    /**
     * @brief objectPoints storage for 3D points of Corners on the target.
     */
    std::vector<std::vector<cv::Point3f> > objectPoints;

    /**
     * @brief cameraMatrix intrinsic parameters of the camera (3x3 homography)
     */
    cv::Mat_<double> cameraMatrix;

    /**
     * @brief distCoeffs distortion coefficients
     */
    cv::Mat_<double> distCoeffs;

    /**
     * @brief stdDevIntrinsics standard deviations of the intrinsiv parameters.
     * From the OpenCV documentation: Order of deviations values: \((f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6 , s_1, s_2, s_3, s_4, \tau_x, \tau_y)\) If one of parameters is not estimated, it's deviation is equals to zero.
     */
    cv::Mat_<double> stdDevIntrinsics;

    /**
     * @brief stdDevExtrinsics standard deviations of the extrinsic parameters (= position and rotation of the targets)
     * From the OpenCV documentation: Order of deviations values: (R1,T1,…,RM,TM) where M is number of pattern views, Ri,Ti are concatenated 1x3 vectors.
     */
    cv::Mat_<double> stdDevExtrinsics;

    /**
     * @brief perViewErrors RMS of the projection error for each view.
     */
    cv::Mat_<double> perViewErrors;

    /**
     * @brief rvecs
     */
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;

    /**
     * @brief imageSize Resolution of the input images.
     */
    cv::Size imageSize;

    /**
     * @brief apertureWidth width of the image sensor in mm
     */
    double apertureWidth = 36; // We are assuming a full frame sensor.

    /**
     * @brief apertureHeight height of the image sensor in mm
     */
    double apertureHeight = 24;


    /**
     * @brief fovx horizontal field of view in degrees.
     */
    double fovx;

    /**
     * @brief fovy vertical field of view in degrees.
     */
    double fovy;

    /**
     * @brief focalLength Focal length in mm.
     */
    double focalLength;

    /**
     * @brief principalPoint in mm
     */
    cv::Point2d principalPoint;

    /**
     * @brief aspectRatio f_y/f_x
     */
    double aspectRatio;

    void plotReprojectionErrors(size_t const ii);

    void plotReprojectionErrors();

    template<class F, class T>
    static void project(
            F const p[3],
    T result[2],
    const T focal[2],
    const T principal[2],
    const T R[9],
    const T t[3]
    );

    template<class F, class T>
    static void project(
            F const p[3],
    T result[2],
    const T focal[2],
    const T principal[2],
    const T R[9],
    const T t[3],
    const T dist[14]
    );

    template<class T>
    static void rot_vec2mat(T const vec[3], T mat[9]);
};

class Calib
{
    bool verbose = true;
    int grid_width = 1;
    int grid_height = 1;

    bool use_rgb = false;

    typedef std::map<std::string, CornerStore> Store_T;
    Store_T data;

    cv::Size imageSize = cv::Size(1000, 1000);

    bool plot_markers = false;

    /**
     * @brief size_known false if the resolution of the input images is not (yet) known.
     */
    bool resolution_known = false;
public:
    Calib();

    static cv::Point3f getInitial3DCoord(hdmarker::Corner const& c, double const z = 0);

    double openCVCalib(CalibrationResult &result);

    void plotMarkers(bool plot = true);

    /**
     * @brief setImageSize set the size of the captured images.
     * @param img
     */
    void setImageSize(cv::Mat const& img);

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
