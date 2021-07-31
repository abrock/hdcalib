#ifndef HDCALIB_H
#define HDCALIB_H

#include <stdio.h>
#include <map>
#include <iostream>
#include <unordered_map>
#include <exception>
#include <thread>

#include <hdmarker/hdmarker.hpp>
#include <hdmarker/subpattern.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <boost/filesystem.hpp>

#include <libraw/libraw.h>

#include <ceres/ceres.h>

#include "nanoflann.hpp"

#include <runningstats/runningstats.h>
#include <catlogger/catlogger.h>


#include <glog/logging.h>

#include "griddescription.h"

#include "randutils.hpp"

#include "cornercache.h"

namespace hdcalib {
using namespace std;
using namespace hdmarker;
using namespace cv;
namespace fs = boost::filesystem;

/**
 * @brief filter_duplicate_markers purges duplicate markers from a vector of markers.
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

class Calib;

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

    size_t last_clean_diff = 0;

public:
    CornerStore();

    void purgeRecursionDeeperThan(int level);

    size_t countMainMarkers() const;

    void sort();

    size_t lastCleanDifference() const;

    std::vector<hdmarker::Corner> getSquaresTopLeft(int const cornerIdFactor, runningstats::QuantileStats<float> *distances = nullptr) const;
    std::vector<hdmarker::Corner> getMainMarkers(int const cornerIdFactor = 10) const;

    std::vector<std::vector<hdmarker::Corner> > getSquares(int const cornerIdFactor, runningstats::QuantileStats<float> *distances = nullptr) const;

    /**
     * @brief CornerStore explicit copy constructor
     * @param c
     */
    CornerStore(const CornerStore& c);

    CornerStore(const std::vector<hdmarker::Corner> & corners);

    /**
     * @brief operator = a better copy operator for avoiding memory problems.
     * @param other
     * @return
     */
    CornerStore& operator=(const CornerStore& other);

    /**
     * @brief clean first calls purgeDuplicates and then purgeUnlikely;
     */
    void clean(int cornerIdFactor);

    /**
     * @brief intersect calculates the intersection between this store and another store
     * and replaces this store's corners by the intersection.
     * @param b
     */
    void intersect(CornerStore const& b);

    static void intersect(CornerStore & a, CornerStore & b);

    /**
     * @brief difference removes all markers in subtrahend from the store.
     * @param subtrahend markers which should be removed.
     */
    void difference(CornerStore const& subtrahend);

    void replaceCorners(std::vector<hdmarker::Corner> const& _corners);

    void scaleIDs(int factor);

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
    std::vector<hdmarker::Corner> findByID(hdmarker::Corner const& ref, size_t const num_results = 1) const;

    /**
     * @brief findByID does a nearest-neighbour search for the num_results closest hdmarker::Corner to a given marker. Distance is the L2 distance of the triples given by (id.x, id.y, page) of the corners.
     * @param ref hdmarker::Corner we are searching.
     * @param num_results maximum number of results we want. The result set might be smaller.
     * @return A vector of results, ordered by distance to ref ascending.
     */
    std::vector<hdmarker::Corner> findByID(cv::Point3i const& ref, size_t const num_results = 1) const;

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
    bool purgeUnlikely(int cornerIdFactor);

    /**
     * @brief purgeDuplicates removes duplicate markers.
     *
     * @return true if duplicates were found.
     */
    bool purgeDuplicates();

    bool purgeOutOfBounds(int const min_x, int const min_y, int const max_x, int const max_y);

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
     * @brief hasID checks if a given hdmarker::Corner (identified by id and page) exists in the CornerStore.
     * @param ref corner we search.
     * @return true if the corner exists.
     */
    bool hasID(hdmarker::Corner const& ref, hdmarker::Corner & found) const;

    /**
     * @brief hasID checks if a given hdmarker::Corner (identified by id and page) exists in the CornerStore.
     * @param ref corner we search.
     * @return true if the corner exists.
     */
    bool hasID(cv::Point3i const& ref, hdmarker::Corner & found) const;

    /**
     * @brief hasIDLevel checks if a given hdmarker::Corner (identified by id, page and level of recursion at which it was detected) exists in the CornerStore.
     * @param ref corner we search.
     * @param level recursion level we require.
     * @return true if the corner exists.
     */
    bool hasIDLevel(hdmarker::Corner const& ref, hdmarker::Corner & found, int8_t level) const;

    /**
     * @brief hasIDLevel checks if a given hdmarker::Corner (identified by id, page and level of recursion at which it was detected) exists in the CornerStore.
     * @param ref corner we search.
     * @param level recursion level we require.
     * @return true if the corner exists.
     */
    bool hasIDLevel(hdmarker::Corner const& ref, int8_t level) const;

    bool hasIDLevel(const cv::Scalar_<int> &id, Corner &found) const;

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

    void getPoints(
            std::vector<cv::Point2f>& imagePoints,
            std::vector<cv::Point3f> & objectPoints,
            hdcalib::Calib const& calib) const;

    void getMajorPoints(std::vector<Point2f> &imagePoints,
                        std::vector<Point3f> &objectPoints,
                        std::vector<cv::Scalar_<int> > &marker_references,
                        hdcalib::Calib const& calib) const;

    void addConditional(const std::vector<Corner> &vec);
    static int getCornerIdFactorFromMainMarkers(const std::vector<Corner> &vec);
    int getCornerIdFactorFromMainMarkers() const;
    static std::map<int, size_t> countLayers(const std::vector<Corner> &vec);
    std::map<int, size_t> countLayers() const;
};

template<int NUM, int DEG>
struct SplineFunctor {
    cv::Vec2f src, dst;
    cv::Size size;
    double factor_x, factor_y;

    static const size_t n = (NUM+DEG)*(NUM+DEG);
    static const size_t n_rows = (NUM+DEG);

    SplineFunctor(cv::Vec2f const& _src, cv::Vec2f const& _dst, cv::Size const& _size);

    cv::Vec2f apply(cv::Vec2f const& pt,
                    cv::Mat_<float> const& weights_x,
                    cv::Mat_<float> const& weights_y) const;

    cv::Vec2f apply(cv::Vec2f const& pt,
                    std::vector<double> const& weights_x,
                    std::vector<double> weights_y) const;

    template<class T>
    bool operator()(T const * const weights_x, T const * const weights_y, T * residuals) const;

    template<class T, class U>
    void apply(T* pt, U const * const weights_x, U const * const weights_y) const;

    template<class T, class U>
    T applySingle(T const * const val, U const * const weights) const;

    template<class T, class U>
    T applyRow(T const& val, U const * const weights) const;
};

template<int LENGTH>
struct VecLengthFunctor {
    double const target_square_length;
    VecLengthFunctor(double const _target_square_length);

    template<class T>
    bool operator() (T const * const vec, T * residual) const;
};

class SimpleProjectionFunctor {
    std::vector<cv::Point2f> const& markers;
    std::vector<cv::Point3f> const& points;

    cv::Point2f const principal;

public:
    SimpleProjectionFunctor(std::vector<cv::Point2f> const& _markers,
                            std::vector<cv::Point3f> const& _points,
                            cv::Point2f const& _principal);
    /*
    1, // focal length f
    3, // rotation vector for the target
    3, // translation vector for the target
    */
    template<class T>
    bool operator()(
            T const* const focal,
            T const* const rvec,
            T const* const tvec,
            T* residuals) const;
};

class ProjectionFunctor {
    std::vector<cv::Point2f> const markers;
    std::vector<cv::Point3f> const points;

public:
    ProjectionFunctor(std::vector<cv::Point2f> const& _markers,
                      std::vector<cv::Point3f> const& _points);
    ~ProjectionFunctor();

    /*
    1, // focal length x
    1, // focal length y
    1, // principal point x
    1, // principal point y
    3, // rotation vector for the target
    3, // translation vector for the target
    14 // distortion coefficients
    */
    template<class T>
    bool operator()(
            T const* const f_x,
            T const* const f_y,
            T const* const c_x,
            T const* const c_y,
            T const* const rvec,
            T const* const tvec,
            T const* const dist,
            T* residuals) const;
};

class ProjectionFunctorRot4 {
    std::vector<cv::Point2f> const& markers;
    std::vector<cv::Point3f> const& points;

public:
    ProjectionFunctorRot4(std::vector<cv::Point2f> const& _markers,
                          std::vector<cv::Point3f> const& _points);
    /*
    1, // focal length x
    1, // focal length y
    1, // principal point x
    1, // principal point y
    3, // rotation vector for the target
    3, // translation vector for the target
    14 // distortion coefficients
    */
    template<class T>
    bool operator()(
            T const* const f_x,
            T const* const f_y,
            T const* const c_x,
            T const* const c_y,
            T const* const rvec,
            T const* const tvec,
            T const* const dist,
            T* residuals) const;
};

class FlexibleTargetProjectionFunctor {
    cv::Point2f const marker;
    cv::Point3f const point;
public:
    double weight = 1;
    FlexibleTargetProjectionFunctor(cv::Point2f const& _marker,
                                    cv::Point3f const& _point);
    /*
    1, // focal length x
    1, // focal length y
    1, // principal point x
    1, // principal point y
    3, // rotation vector for the target
    3, // translation vector for the target
    3, // correction vector for the 3d marker position
    14 // distortion coefficients
    */
    template<class T>
    bool operator()(
            T const* const f_x,
            T const* const f_y,
            T const* const c_x,
            T const* const c_y,
            T const* const rvec,
            T const* const tvec,
            T const* const correction,
            T const* const dist,
            T* residuals) const;
};

template<int N>
class FlexibleTargetProjectionFunctorN {
    cv::Point2f const marker;
    cv::Point3f const point;
public:
    double weight = 1;
    FlexibleTargetProjectionFunctorN(cv::Point2f const& _marker,
                                    cv::Point3f const& _point);
    /*
    1, // focal length x
    1, // focal length y
    1, // principal point x
    1, // principal point y
    3, // rotation vector for the target
    3, // translation vector for the target
    3, // correction vector for the 3d marker position
    14 // distortion coefficients
    */
    template<class T>
    bool operator()(
            T const* const f_x,
            T const* const f_y,
            T const* const c_x,
            T const* const c_y,
            T const* const rvec,
            T const* const tvec,
            T const* const correction,
            T const* const dist,
            T* residuals) const;
};

template<int N>
class FlexibleTargetProjectionFunctorOdd {
    cv::Point2f const marker;
    cv::Point3f const point;
public:
    double weight = 1;
    FlexibleTargetProjectionFunctorOdd(cv::Point2f const& _marker,
                                    cv::Point3f const& _point);
    /*
    1, // focal length x
    1, // focal length y
    1, // principal point x
    1, // principal point y
    3, // rotation vector for the target
    3, // translation vector for the target
    3, // correction vector for the 3d marker position
    14 // distortion coefficients
    */
    template<class T>
    bool operator()(
            T const* const f_x,
            T const* const f_y,
            T const* const c_x,
            T const* const c_y,
            T const* const rvec,
            T const* const tvec,
            T const* const correction,
            T const* const x_factor,
            T const* const dist,
            T* residuals) const;
};

class SingleProjectionFunctor {
    cv::Point2f const marker;
    cv::Point3f const point;
public:
    double weight = 1;
    SingleProjectionFunctor(cv::Point2f const& _marker,
                            cv::Point3f const& _point);
    /*
    1, // focal length x
    1, // focal length y
    1, // principal point x
    1, // principal point y
    3, // rotation vector for the target
    3, // translation vector for the target
    3, // correction vector for the 3d marker position
    14 // distortion coefficients
    */
    template<class T>
    bool operator()(
            T const* const f_x,
            T const* const f_y,
            T const* const c_x,
            T const* const c_y,
            T const* const rvec,
            T const* const tvec,
            T const* const dist,
            T* residuals) const;
};

struct DistortedTargetCorrectionFunctor {
    cv::Point3f const src;
    cv::Point2f const center;

    DistortedTargetCorrectionFunctor(cv::Point3f const& _src, cv::Point2f const& _center);
    /*
    3, // correction vector for the 3d marker position
    14 // distortion coefficients
    */
    template<class T, class U>
    bool operator()(
            T const* const correction,
            U const* const inverse_dist,
            T* residuals) const;
};

class SemiFlexibleTargetProjectionFunctor {
    cv::Point2f const marker;
    cv::Point3f const point;
    cv::Point2f const center;
public:
    double weight = 1;
    SemiFlexibleTargetProjectionFunctor(cv::Point2f const& _marker,
                                        cv::Point3f const& _point,
                                        cv::Point2f const& _center);
    /*
    1, // focal length x
    1, // focal length y
    1, // principal point x
    1, // principal point y
    3, // rotation vector for the target
    3, // translation vector for the target
    3, // correction vector for the 3d marker position
    14 // distortion coefficients
    */
    template<class T>
    bool operator()(
            T const* const f_x,
            T const* const f_y,
            T const* const c_x,
            T const* const c_y,
            T const* const rvec,
            T const* const tvec,
            T const* const correction,
            T const* const dist,
            T const* const inverse_dist,
            T* residuals) const;

    template<class T, class U>
    static void applyInverseDist(const T src[], T dst[], const T center[], const U dist[]);
};


template<class C>
struct cmpSimpleIndex3 {
    bool operator()(const C& a, const C& b) const;
};

struct cmpPoint3i {
    bool operator()(const cv::Point3i &a, const cv::Point3i &b) const;
};

struct cmpScalar {
    template<class T>
    bool operator()(const cv::Scalar_<T> &a, const cv::Scalar_<T> b) const;
};

class CalibResult {
public:
    Calib * calib;
    /**
     * @brief objectPointCorrections Corrections to the objectPoints
     * which allow the optimization to account for systematic misplacements
     * by the marker detection algorithm
     */
    std::map<cv::Scalar_<int>, cv::Point3f, cmpScalar> objectPointCorrections;

    std::map<cv::Scalar_<int>, cv::Point3f, cmpScalar> raw_objectPointCorrections;

    /**
     * @brief cameraMatrix intrinsic parameters of the camera (3x3 homography)
     */
    cv::Mat_<double> cameraMatrix;

    std::vector<double> outlier_percentages;

    /**
     * @brief distCoeffs distortion coefficients
     * From the OpenCV documentation: Order of deviations values: \((f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6 , s_1, s_2, s_3, s_4, \tau_x, \tau_y)\) If one of parameters is not estimated, it's deviation is equals to zero.
     */
    cv::Mat_<double> distCoeffs;

    /**
     * @brief dist12 distortion coefficients for 16-parameter rational function
     */
    std::vector<double> distN;

    /**
     * @brief spline_x vector of coefficients for the spline interpolation method
     */
    std::vector<double> spline_x;
    std::vector<double> spline_y;

    /**
     * @brief x_factor percentage of the calibration target non-uniformity (x-scale being different from y-scale)
     */
    double x_factor = 0;

    /**
     * @brief error_quantiles is a vector containing 101 elements, one for each error percentile.
     */
    std::vector<double> error_percentiles;

    double error_median = -1;

    std::vector<double> getErrorPercentiles();

    /**
     * @brief distCoeffs distortion coefficients of the estimated distortion of the target. This prevents the calibration
     * from trading real distortion for non-sense distortion of the target.
     * From the OpenCV documentation: Order of deviations values: \((f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6 , s_1, s_2, s_3, s_4, \tau_x, \tau_y)\) If one of parameters is not estimated, it's deviation is equals to zero.
     */
    std::vector<double> inverseDistCoeffs;

    /**
     * @brief stdDevIntrinsics standard deviations of the intrinsic parameters.
     */
    cv::Mat_<double> stdDevIntrinsics;

    /**
     * @brief stdDevIntrinsics standard deviations of the new object points estimated by calibrateCameraRO
     */
    cv::Mat_<double> stdDevObjectPoints;

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

    std::vector<std::string> imageFiles;

    /**
     * @brief undistortion Map for undistortion and rectification at the same time.
     */
    cv::Mat_<cv::Vec2f> undistortRectifyMap;

    /**
     * @brief rectification Rotation vector needed for rectifying light field images.
     */
    cv::Mat_<double> rectification;

    CalibResult() {}

    CalibResult(const CalibResult &t) {
        *this = t;
    }

    CalibResult& operator = (const CalibResult &t)
    {
        objectPointCorrections = t.objectPointCorrections;
        cameraMatrix = t.cameraMatrix.clone();
        distCoeffs = t.distCoeffs.clone();
        distN = t.distN;
        name = t.name;
        x_factor = t.x_factor;
        errors = t.errors;
        error_percentiles = t.error_percentiles;
        stdDevIntrinsics = t.stdDevIntrinsics.clone();
        stdDevExtrinsics = t.stdDevExtrinsics.clone();
        perViewErrors = t.perViewErrors.clone();
        rvecs = t.rvecs;
        tvecs = t.tvecs;
        imageFiles = t.imageFiles;
        undistortRectifyMap = t.undistortRectifyMap.clone();
        rectification = t.rectification.clone();
        return *this;
    }

    void keepMarkers(const CornerStore &keep);

    void write(cv::FileStorage & fs) const;
    void read(const FileNode &node);

    /**
     * @brief isExpandedType checks if the calibration uses an expanded Model,
     * those are named (Semi)Flexible(Odd|N)
     * @return
     */
    bool isExpandedType();

    /**
     * @brief getTVec returns the translation vector corresponding to a given filename.
     * @param filename
     * @return
     */
    Mat getTVec(const string &filename) const;

    /**
     * @brief getRVec returns the rotation vector corresponding to a given filename.
     * @param filename
     * @return
     */
    cv::Mat getRVec(std::string const& filename) const;

    bool is_valid = false;

    /**
     * @brief name Name of the calibration result, e.g. "SimpleOpenCV", "OpenCV", "Ceres", "Flexible", "SimpleCeres"
     */
    std::string name;

    void scaleResult(double const ratio);
    template<class F, class T>
    void projectByCalibName(
            const F p[],
            T result[],
            const T focal[],
            const T principal[],
            const T R[],
            const T t[]);
    std::vector<double> getDistCoeffsVector();

    void getReprojections(
            const size_t ii,
            std::vector<Point2d> &markers,
            std::vector<Point2d> &reprojections);

    void getAllReprojections(
            std::vector<Point2d> &markers,
            std::vector<Point2d> &reprojections);

    double getErrorMedian();

    runningstats::QuantileStats<float> errors;
}; // class CalibResult

void write(cv::FileStorage& fs, const std::string&, const CalibResult& x);
void read(const cv::FileNode& node, CalibResult& x, const CalibResult& default_value = CalibResult());


class Calib {
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

    std::vector<std::vector<cv::Scalar_<int> > > reduced_marker_references;

    /**
     * @brief third_fixed_point_index is the index of the third point to be fixed in cv::calibrateCameraRO
     */
    size_t third_fixed_point_index = 0;

    /**
     * @brief openCVMaxPoints Maximum number of 2D markers per image in the OpenCV calibration.
     */
    size_t openCVMaxPoints = 1000;

    /**
     * @brief max_iter Maximum number of iterations used in the calibrations via Ceres.
     */
    size_t max_iter = 1000;

    static void printObjectPointCorrectionsStats(std::string const& name,
                                          const std::map<cv::Scalar_<int>, Point3f, cmpScalar> &corrections);

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

    bool verbose = true;
    bool verbose2 = false;

    int grid_width = 1;
    int grid_height = 1;

    bool use_rgb = false;

    bool use_raw = false;

    bool plotMarkers = false;

    bool plotSubMarkers = false;

    /**
     * @brief size_known false if the resolution of the input images is not (yet) known.
     */
    bool resolutionKnown = false;

    /**
     * @brief white_balance Color of white (BGR)
     */
    cv::Point3f white_balance = cv::Point3f(200,170,80);

    /**
     * @brief use_only_green If set to true only the green channel will be used.
     */
    bool useOnlyGreen = false;

    static char color(int const ii, int const jj);

    /**
     * @brief cornerIdFactor Scale factor for the corner IDs when using submarkers.
     * Without submarkers corner id values are in [0...32].
     */
    int cornerIdFactor = 1;

    /**
     * @brief recursionDepth Recursion depth for the marker detection.
     */
    int recursionDepth = 1;

    /**
     * @brief markerSize Physical size of the major markers on the target in any unit.
     */
    double markerSize = 6.35;

    /**
     * @brief validPages page numbers used in the calibration target, corners with different page numbers will be considered false detections.
     */
    std::vector<int> validPages = {6,7};

    bool preparedOpenCVCalib = false;
    bool preparedOpenCVROCalib = false;
    bool preparedCalib = false;

    std::vector<cv::Scalar> const color_circle = {
        cv::Scalar(255,255,255,255),
        cv::Scalar(255,0,0,255),
        cv::Scalar(0,255,0,255),
        cv::Scalar(0,0,255,255),
        cv::Scalar(255,255,0,255),
        cv::Scalar(0,255,255,255),
        cv::Scalar(255,0,255,255)
    };

    bool hasCalibration = false;

    /**
     * @brief calibrations Map of calibrations calculated. Those keys are used:
     * "SimpleOpenCV" -> simple calibration using the OpenCV calibration method ("calibrateCamer") but using only target positions and focal length as free parameters,
     * no distortion and no central pixel estimation.
     * "OpenCV" -> Initial calibration using the OpenCV calibration method ("calibrateCamera").
     * "Ceres" -> Refinement using Ceres.
     * "Flexible" -> Refinement using Ceres and the flexible target model
     * "SemiFlexible" -> Using a semi-flexible model where not each individual marker
     * may have an offset but instead each type (two corner types and black/white markers
     * depending on recursion level)
     */
    std::map<std::string, CalibResult> calibrations;

    /**
     * @brief max_outlier_percentage Maximum percentage of outliers for an image to be included in the Ceres calibration methods.
     */
    double max_outlier_percentage = 105;

    double cauchy_param;

    bool demosaic = false;
    bool libraw = false;
    double effort = 0.5;
    double outlier_threshold = 5;

    double min_snr = 5;

    double x_factor = 0;

public:
    typedef std::map<std::string, CornerStore> Store_T;
    Store_T data;

    void purgeRecursionDeeperThan(int level);

    void autoScaleCornerIds();

    static size_t countMainMarkers(std::vector<Corner> const& vec);

    void prepareCalibrationByName(std::string const& name);

    void combineImages(std::string const& out_file);

    static std::vector<double> mat2vec(cv::Mat_<double> const& m);
    static cv::Mat_<double> vec2squaremat(std::vector<double> const& vec);

    static void scaleSquareMatVec(std::vector<double> & vec, const int N);

    Calib();

    unsigned int threads = std::thread::hardware_concurrency() > 0 ? std::thread::hardware_concurrency() : 4;

    void save(std::string const& filename);

    void setMinSNR(double const val);

    void setUseRaw(bool const val);

    void setMaxIter(size_t const val);

    /**
     * @brief getImageNames returns a (sorted) list of all filenames stored in the data map.
     * @return
     */
    std::vector<std::string> getImageNames() const;

    /**
     * @brief purgeSubmarkers removes all submarkers from the data so only the main markers remain.
     */
    void purgeSubmarkers();

    static cv::Point3f getTransformedPoint(CalibResult& res, std::string const& filename, cv::Point3f const& pt);

    void checkSamePosition(std::vector<std::string> const& suffixes, std::string const calibration_type = "Flexible");
    void checkSamePosition2D(std::vector<std::string> const& suffixes);

    std::vector<std::vector<cv::Vec3d> > getCommon3DPoints(CalibResult & calib, std::vector<std::string> const& files);

    static std::map<std::string, std::vector<std::string> > matchSuffixes(std::vector<std::string> const& images, std::vector<std::string> const& suffixes);

    std::string printAllCameraMatrices();

    void setMaxOutlierPercentage(double const new_val);

    void deleteCalib(std::string const name);

    void setCauchyParam(double const new_val);

    void plotResidualsIntoImages(std::string const calib_name);

    void setOutlierThreshold(double const new_val);

    void purgeUnlikelyByDetectedRectangles();

    /**
     * @brief getIdRectangleUnion Finds the rectangle containing all main marker ids
     * @return
     */
    Rect_<int> getIdRectangleUnion() const;


    static double distance(hdmarker::Corner const& a, hdmarker::Corner const& b);

    cv::Mat_<uint8_t> getMainMarkersArea(std::vector<hdmarker::Corner> const& submarkers, const Scalar color = cv::Scalar::all(255), const int line = cv::LINE_AA);

    void exportPointClouds(std::string const& calib_name, const double outlier_threshold = -1);

    /**
     * @brief calculateUndistortion calculates the undistortion map using cv::initUndistortRectifyMap from OpenCV.
     * @return
     */
    cv::Mat calculateUndistortRectifyMap(CalibResult &calib);

    /**
     * @brief getCachedUndistortRectifyMap returns the cached undistortRectifyMap and creates it if neccessary.
     * @return
     */
    cv::Mat getCachedUndistortRectifyMap(const string &calibName);

    template<class T, class U>
    static void normalize_rot4(T const in[4], U out[4]);

    static void normalizeRotationVector(cv::Mat & vector);

    static void normalizeRotationVector(double vector[3]);

    double getMarkerSize() const;

    void invalidateCache();

    int getCornerIdFactor() const;

    static int computeCornerIdFactor(int const recursion_depth);

    void setValidPages(std::vector<int> const& _pages);

    /**
     * @brief purgeInvalidPages Remove corners if the page number is not in the validPages vector.
     */
    void purgeInvalidPages();

    static std::vector<hdmarker::Corner> purgeInvalidPages(std::vector<hdmarker::Corner> const& in, std::vector<int> const& valid_pages);

    static bool isValidPage(int const page, std::vector<int> const& valid_pages);
    bool isValidPage(int const page) const;
    bool isValidPage(hdmarker::Corner const& c) const;

    void setRecursionDepth(const int _recursionDepth);

    typedef std::map<cv::Scalar_<int>, std::vector<std::pair<cv::Point2f, cv::Point2f> >, cmpScalar> MarkerMap;

    static void scaleCornerIds(std::vector<hdmarker::Corner>& corners, int factor);

    static void piecewiseRefinement(cv::Mat &img, std::vector<hdmarker::Corner> const& in, std::vector<hdmarker::Corner> & out, int recursion_depth, double & markerSize);

    static void refineRecursiveByPage(
            cv::Mat &img,
            std::vector<hdmarker::Corner> const& in,
            std::vector<hdmarker::Corner> & out,
            const int recursion_depth,
            double & markerSize);

    /**
     * @brief prepareCalibration fills the containers imagePoints, objectPoints and imageFiles.
     */
    void prepareCalibration();

    /**
     * @brief prepareOpenCVCalibration Prepare calibration data for the OpenCV calibration routine.
     * It uses way too much memory per correspondence, therefore a subset of 1000 markers per input image is used.
     */
    void prepareOpenCVCalibration();

    /**
     * @brief hasFile checks if the given file is already known to the class.
     * @param filename
     * @return
     */
    bool hasFile(std::string const filename) const;

    size_t getId(std::string const& filename) const;

    /**
     * @brief removeOutliers Removes outliers from the detected markers identified by having a reprojection error larger than some threshold.
     * @param threshold error threshold in px.
     */
    bool removeOutliers(const string &calibName, double const threshold = 2);

    /**
     * @brief removeOutliers calls removeOutliers for each stored calibration result.
     * @param threshold error threshold in px.
     */
    bool removeAllOutliers(double const threshold = 2);

    void only_green(bool only_green = true);

    template<class Point>
    static double distance(Point const a, Point const b);

    static std::vector<double> mat2vec(cv::Mat const& in);

    static cv::Mat_<double> vec2mat(std::vector<double> const& in);

    static std::vector<double> point2vec3f(cv::Point3f const& in);

    static cv::Point3f vec2point3f(std::vector<double> const& in);

    static void white_balance_inplace(cv::Mat & mat, const Point3f white);

    /**
     * @brief plotReprojectionErrors plots all reprojection errors of all input images.
     * @param prefix prefix for all files
     * @param suffix suffix for all files (before filename extension)
     */
    void plotReprojectionErrors(const string &calibName, const string prefix = "",
                                const std::string suffix ="");

    /**
     * @brief plotReprojectionErrors plots reprojection errors of a sinle input image.
     * @param ii index of the input image.
     * @param prefix prefix for all files.
     * @param suffix suffix for all files (before filename extension)
     */
    void plotReprojectionErrors(const std::string &calibName,
                                const size_t image_index,
                                MarkerMap &residuals_by_marker,
                                const std::string prefix,
                                const std::string suffix,
                                std::vector<float> &res_x,
                                std::vector<float> &res_y);


    void plotErrorsByMarker(MarkerMap const& map,
                            const std::string prefix="",
                            const std::string suffix="");


    void plotResidualsByMarkerStats(MarkerMap const& map,
                                    const std::string prefix="",
                                    const std::string suffix="");

    /**
     * @brief meanResidual calculate mean residual given a vector of pairs of detected markers and reprojections.
     * @param data
     * @return
     */
    static cv::Point2f meanResidual(std::vector<std::pair<cv::Point2f, cv::Point2f> > const& data);

    /**
     * @brief getSimpleId returns the "id" of the marker which consists of the marker's id value and page number in one single cv::Point3i for easy usage in std::map etc.
     * @param marker
     * @return
     */
    static cv::Point3i getSimpleId(hdmarker::Corner const & marker);

    /**
     * @brief getSimpleId returns the "id" of the marker which consists of the marker's id value, page number and layer
     * in one single cv::Scalar_<int> for easy usage in std::map etc.
     * @param marker
     * @return
     */
    static cv::Scalar_<int> getSimpleIdLayer(const Corner &marker);

    static uint64_t getIdHash(hdmarker::Corner const& marker);

    /**
     * @brief findOutliers finds outliers of the detected markers with a reprojection error above some threshold in one of the images.
     * @param threshold error threshold in px.
     * @param ii index of the image to search for outliers in.
     * @param outliers vector of detected outliers
     */
    void findOutliers(const string &calib_name,
                      double const threshold,
                      size_t const ii,
                      std::vector<Corner> &outliers);

    template<class F, class T>
    static void project(
            F const p[3],
    T result[2],
    const T focal[2],
    const T principal[2],
    const T R[9],
    const T t[3]
    );

    cv::Point2f project(const cv::Mat_<double> &cameraMatrix, cv::Vec3d const& point) const;

    template<class F, class T>
    /**
     * @brief project
     * @param name Name of the calibration result.
     * @param p 3D world point
     * @param result 2D projected point
     * @param focal focal lengths (f_x and f_y)
     * @param principal Principal point (orthogonal projection of the pinhole on the sensor)
     * @param R Rotation matrix (3x3)
     * @param t translation vector
     * @param dist distortion vector
     */
    static void projectByCalibName(
    std::string const& name,
    F const p[3],
    T result[2],
    const T focal[2],
    const T principal[2],
    const T R[9],
    const T t[3],
    const T dist[]
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

    template<int N, class F, class T>
    static void projectN(
            F const p[3],
    T result[2],
    const T focal[2],
    const T principal[2],
    const T R[9],
    const T t[3],
    const T dist[N+8]
    );

    template<int N, class F, class T>
    static void projectOdd(
            F const p[3],
    T result[2],
    const T focal[2],
    const T principal[2],
    const T R[9],
    const T t[3],
    const T dist[N+8]
    );

    template<class F, class T>
    static void get3DPoint(
            F const p[3],
    T result[3],
    const T R[9],
    const T t[3]
    );

    /**
     * @brief get3DPoint calculates the 3D point of a hdmarker::Corner given a rotation and translation vector.
     * It takes into account the correction of the 3D point.
     * @param c
     * @param _rvec
     * @param _tvec
     * @return
     */
    cv::Vec3d get3DPoint(CalibResult & calib, hdmarker::Corner const& c, cv::Mat const& _rvec, cv::Mat const& _tvec);

    template<class T>
    /**
     * @brief rot_vec2mat converts a Rodriguez rotation vector (with 3 entries) to a 3x3 rotation matrix.
     * @param vec
     * @param mat
     */
    static void rot_vec2mat(T const vec[3], T mat[9]);

    template<class T>
    /**
     * @brief rot4_vec2mat converts a 4-entry rotation vector (first three entries describe the axis, the fourth the amount of rotation)
     * into a 3x3 rotation matrix. This avoids the case analysis of the 3-entry Rodriguez vector which
     * @param vec
     * @param mat
     */
    static void rot4_vec2mat(T const vec[4], T mat[9]);

    template<class T>
    /**
     * @brief rot3_rot4 Converts a (3-entry) Rodrigues rotation vector to a equivalent rotation vector with 4 components:
     * The first three denote the rotation axis while the fourth gives the amount of rotation.
     * @param rvec
     * @param vec
     */
    static void rot3_rot4(cv::Mat const& rvec, T vec[4]);

    template<class T>
    /**
     * @brief rot3_rot4 Converts a (3-entry) Rodrigues rotation vector to a equivalent rotation vector with 4 components:
     * The first three denote the rotation axis while the fourth gives the amount of rotation.
     * @param rvec
     * @param vec
     */
    static void rot3_rot4(T const src[3], T vec[4]);

    template<class T>
    static std::vector<T> rot3_rot4(cv::Mat const& rvec);

    template<class T>
    /**
     * @brief rot4_rot3 Converts a (4-entry) rotation vector to a equivalent Rodrigues rotation vector with 3 components.
     * @param rvec
     * @param vec
     */
    static void rot4_rot3(T const vec[4], Mat &rvec);

    template<class T>
    /**
     * @brief rot4_rot3 Converts a (4-entry) rotation vector to a equivalent Rodrigues rotation vector with 3 components.
     * @param rvec
     * @param vec
     */
    static void rot4_rot3(T const vec[4], T rvec[3]);

    template<class T>
    /**
     * @brief rot_vec2mat converts a Rodriguez rotation vector (with 3 entries) to a 3x3 rotation matrix.
     * @param vec
     * @param mat
     */
    static void rot_vec2mat(cv::Mat const& vec, T mat[9]);

    static std::string tostringLZ(size_t num, size_t min_digits = 2);

    /**
     * @brief getRectificationRotation Find the rotation vector in object space needed for the rectification of the captured light field images.
     * This means: Find one rotation R such that when R is applied to the estimated 3D locations of the markers on the calibration target
     * the projected markers of images in the same row only move along the x axis when projected using the camera model without distortion;
     * and projected markers of images in the same columns only move along the y axis.
     * @param rows
     * @param cols
     * @param images
     * @param rect_rot
     */
    void getRectificationRotation(CalibResult &calib, size_t const rows, size_t const cols, std::vector<std::string> const& images, cv::Vec3d & rect_rot);

    bool hasCalibName(std::string const& name) const;

    cv::Point3f getInitial3DCoord(hdmarker::Corner const& c, double const z = 0) const;

    cv::Point3f getInitial3DCoord(cv::Point3i const& c, double const z = 0) const;

    void setMarkerSize(const double size);

    /**
     * @brief openCVCalib Use OpenCV Calibration to get initial guess.
     * @param simple Set this to true if no distortion or central pixel should be estimated, just target positions and focal length.
     * @return
     */
    double openCVCalib(const bool simple = false, const bool RO = false);

    double runCalib(std::string const name, double const outlier_threshold = -1);

    /**
     * @brief CeresCalib run the OpenCV calibration re-implemented using Ceres.
     * @param outlier_threshold maximum distance between marker and reprojection when building the problem.
     * Set this to a negative number for the first run in order to make the function ignore it
     * and to a positive number (e.g. 2(px)) for subsequent runs to ignore outliers.
     * @return
     */
    double CeresCalib(double const outlier_threshold = -1);

    double CeresCalibRot4();

    /**
     * @brief CeresCalibFlexibleTarget calibrates the camera using a bundle adjustment
     * algorithm where the exact locations of the markers are free variables.
     * @return
     */
    double CeresCalibFlexibleTarget(const double outlier_threshold = -1);

    void setPlotMarkers(bool plot = true);
    void setPlotSubMarkers(bool plot = true);

    /**
     * @brief setImageSize set the size of the captured images.
     * @param img
     */
    void setImageSize(cv::Mat const& img);

    /**
     * @brief setImageSize set the size of the captured images.
     * @param img
     */
    void setImageSize(cv::Size const& s);

    cv::Size getImageSize() const;

    /**
     * @brief keepCommonCorners Removes all corners from the data which are not present
     * in all detected images. This is needed for the OpenCV calibration which assumes
     * checkerboard-like targets where the whole target is visible in each image.
     */
    void keepCommonCorners_delete();

    void keepCommonCorners_intersect();

    void keepCommonCorners();

    void addInputImage(std::string const filename, std::vector<hdmarker::Corner> const& corners);

    /**
     * @brief addInputImageAfterwards Adds an input image to the dataset after the initial calibration is finished.
     * The rvec and tvec for the calibration target are estimated via solvePNP
     * @param filename
     * @param corners
     */
    void addInputImageAfterwards(std::string const filename, std::vector<hdmarker::Corner> const& corners);

    void addInputImage(std::string const filename, CornerStore const& corners);

    /**
     * @brief addInputImage Adds an input image where rvec and tvec are already known.
     * @param filename filename of the input image.
     * @param corners Corners vector with all detected corners.
     * @param rvec Rotation vector of the calibration target.
     * @param tvec Translation vector of the calibration target.
     */
    void addInputImage(const string filename, const std::vector<Corner> &corners, cv::Mat const& rvec, cv::Mat const& tvec);

    void removeInputImage(const string filename);

    CornerStore get(std::string const filename) const;

    cv::Mat normalize_raw_per_channel(cv::Mat const& input);

    void normalize_raw_per_channel_inplace(cv::Mat & input);

    static cv::Mat read_raw(std::string const& filename);

    CornerStore getUnion() const;

    /**
   * @brief getCorners Tries to read detected hdmarker corners from cache files and if that fails reads the image
   * and runs marker detection.
   * @param input_file
   * @param effort
   * @param demosaic
   * @param recursion_depth
   * @param raw
   * @return
   */
    vector<Corner> getCorners(const std::string input_file,
                              const float effort,
                              const bool demosaic,
                              const bool raw);

    static Mat readImage(std::string const& input_file,
                         bool const demosaic,
                         bool const raw,
                         bool const useOnlyGreen);

    cv::Size read_raw_imagesize(const string &filename);

    void printObjectPointCorrectionsStats(const string &calibName);

    /**
     * @brief write Function needed for serializating a Corner using the OpenCV FileStorage system.
     * @param fs
     */
    void write(cv::FileStorage& fs) const;

    /**
     * @brief read Method needed for reading a serialized Corner using the OpenCV FileStorage system.
     * @param node
     */
    void read(const cv::FileNode& node);

    template<class T, class T1, class T2>
    /**
     * @brief insertSorted Sorts vector a using the default comparison operator of T assuming that the vector is already sorted except for the last element.
     * The vectors b and c are simultaneously updated so the relationship between elements of a, b and c are maintained.
     * @param a primary vector used for sorting.
     * @param b
     * @param c
     */
    static void insertSorted(std::vector<T> &a, std::vector<T1> &b, std::vector<T2> &c);

    void analyzeGridLF(std::string const calibName,
                       size_t const rows,
                       size_t const cols,
                       std::vector<std::string> const& images);

    void getGridVectors(CalibResult &calib,
                        size_t const rows,
                        size_t const cols,
                        std::vector<std::string> const& images,
                        cv::Vec3d & row_vec,
                        cv::Vec3d & col_vec);

    void printHist(std::ostream &out, const runningstats::Histogram &h, const double threshold = 0);
    void getGridVectors2(CalibResult &calib, const size_t rows, const size_t cols, const std::vector<string> &images, Vec3d &row_vec, Vec3d &col_vec);
    void getIndividualRectificationRotation(CalibResult &calib, const size_t rows, const size_t cols, const std::vector<std::string> &images, cv::Vec3d &rect_rot);
    void paintSubmarkers(const std::vector<Corner> &submarkers, cv::Mat &image, int const paint_size_factor) const;

    /**
     * @brief paintSubmarkersRMS_SNR plots float-images of RMS and SNR.
     * @param prefix
     * @param submarkers
     */
    void paintSubmarkersRMS_SNR(
            const std::string& prefix,
            const std::vector<Corner>& submarkers,
            const Size size,
            const int paint_size_factor) const;

    void initializeCameraMatrix(double const focal_length, double const cx, double const cy);
    void initialzeDistortionCoefficients();

    CalibResult & getCalib(const std::string& name);

    template<class Point>
    static bool validPixel(Point const& p, cv::Size const& image_size);

    struct RectifyCost {
        int8_t const axis;
        cv::Vec3d const src_a;
        cv::Vec3d const src_b;

        cv::Mat_<double> const& cameraMatrix;

        float const weight;

        RectifyCost(int8_t const _axis,
                    cv::Vec3d const _src_a,
                    cv::Vec3d const _src_b,
                    cv::Mat_<double> const& _cameraMatrix,
                    float const _weight = 1);

        template<class T>
        void compute(
                T const * const rot_vec,
                T * result1,
                T * result2
                ) const;

        template<class T>
        bool operator () (
                T const * const rot_vec,
                T * residuals) const;
    };

    template<class T>
    /**
     * @brief lensfunDistortionModel
     * @param a
     * @param b
     * @param c
     * @param r
     */
    T lensfunDistortionModel(T const & a, T const& b, T const& c, T const& r);

    void rectificationResidualsPlotsAndStats(
            const char *log_name,
            const std::map<std::string, std::vector<RectifyCost *> > &cost_functions,
            double rot_vec[3],
    bool plot);

    /**
     * @brief readCorners reads hdmarker::Corner objects from a cache file.
     * @param input_file
     * @return
     */
    static std::vector<hdmarker::Corner> readCorners(const std::string &input_file,
                                                     int & width,
                                                     int & height);

    /**
     * @brief readCorners reads hdmarker::Corner objects from a cache file.
     * @param input_file
     * @return
     */
    static std::vector<hdmarker::Corner> readCorners(const std::string &input_file);
    Vec3d get3DPointWithoutCorrection(const Corner &c, const Mat &_rvec, const Mat &_tvec);
    void plotPoly(cv::Mat &img, const std::vector<cv::Point> &poly, const cv::Scalar &color, const int line);
    double SimpleCeresCalib(double const outlier_threshold = -1);

    void setCeresTolerance(double const new_tol);
    static Vec3d get3DPointWithoutCorrection(const cv::Point3f &_src, const Mat &_rvec, const Mat &_tvec);

    double ceres_tolerance = 1e-10;

    vector<Corner> getSubMarkers(const std::string input_file, const float effort = 0.5, const bool demosaic=false, const bool raw=false, bool *is_clean = nullptr);
    vector<Corner> getMainMarkers(const std::string input_file, const float effort, const bool demosaic, const bool raw);
    cv::Mat getImageScaled(const string &input_file);
    static cv::Mat scaleImage(const cv::Mat &img);
    static void printCornerStats(const std::vector<Corner> &vec);
    cv::Mat getImageRaw(const std::string &input_file);
    cv::Mat convert16_8(const cv::Mat &img);

    /**
     * @brief swapPointsForRO Changes the order of the points so obj[0] is the top left, obj[1] the top right and obj.back() the bottom right point.
     * @param img Image points (2D coordinates of detected markers)
     * @param obj Object points (Initial 3D coordinates of the target)
     * @param ref Reference indices for finding the corresponding points in the CornerStore
     */
    static void swapPointsForRO(std::vector<cv::Point2f> &img, std::vector<cv::Point3f> &obj, std::vector<size_t> &ref);

    /**
     * @brief filterSNR Takes a vector of Corners and returns a vector of only those Corners
     * where the SNR*sigma is above the threshold (and all the main markers).
     * @param in
     * @param threshold
     * @return
     */
    vector<Corner> filterSNR(const vector<Corner> &in, const double threshold);

    /**
     * @brief prepareOpenCVROCalibration prepares calibration for calibrateCameraRO since this one requires all points to be visible in all images.
     */
    void prepareOpenCVROCalibration();

    /**
     * @brief plotObjectPointCorrections plots objectPointCorrection into optical-flow files
     * @param calibName
     * @param prefix
     * @param suffix
     */
    void plotObjectPointCorrections(const std::map<cv::Scalar_<int>, Point3f, cmpScalar> &data, const std::string &calibName, string prefix, const string suffix);

    /**
     * @brief plotObjectPointCorrections plots objectPointCorrection into optical-flow files
     * @param calibName
     * @param prefix
     * @param suffix
     */
    void plotObjectPointCorrections(const std::string &calibName, string prefix, const string suffix);

    template<class T>
    /**
     * @brief isValidValue Checks if a value (float, double, cv::Vec2f) is finite and within a given threshold.
     * @param val
     * @param threshold
     * @return
     */
    static bool isValidValue(const T &val, const double threshold);

    template<class T>
    /**
     * @brief fillHoles Fills holes in an image with mean values of neighbours
     * @param _src
     * @return
     */
    cv::Mat_<T> fillHoles(const cv::Mat_<T> &_src, const int max_tries = -1);
    double CeresCalibSemiFlexibleTarget(const double outlier_threshold);
    void minMaxPoint3f(const cv::Point3f &x, cv::Point3f &min, cv::Point3f &max);
    Point3f getInitial3DCoord(const cv::Scalar &c, const double z = 0) const;
    static std::string printVec(const std::vector<double> &vec);
    double CeresCalibKnownCorrections(const double outlier_threshold, CalibResult &calib);
    void checkHDMPointcache(const std::string &input_file, const std::vector<Corner> &corners);
    void plotMeanResiduals(const MarkerMap &data, string prefix, const string suffix, const double pre_filter = -1);
    static int tolerantGCD(int a, int b);
    template<int N>
    double CeresCalibFlexibleTargetN(const double outlier_threshold);
    template<int N>
    double CeresCalibFlexibleTargetOdd(const double outlier_threshold);
    static bool startsWith(const std::string &str, const std::string &term);
    std::vector<std::vector<cv::Point2f> > getImagePoints() const;

    std::vector<std::vector<cv::Point3f> > getObjectPoints() const;

    int getImageIndex(const string &filename);
    template<int NUM, int DEG>
    double CeresCalibFlexibleTargetSpline(const double outlier_threshold);

    template<int NUM, int DEG, class F, class T>
    static void projectSpline(const F p[], T result[], const T focal[], const T principal[], const T R[], const T t[], const T weights_x[], const T weights_y[], const cv::Size &size);
    template<class T>
    static T evaluateSpline(const T x, const int POS, const int DEG);
private:
    template<class RCOST>
    void addImagePairToRectificationProblem(
            CalibResult & calib,
            const CornerStore &current,
            const size_t current_id,
            const CornerStore &next,
            const size_t next_id,
            std::vector<RCOST *> &target_costs,
            ceres::Problem &problem,
            const int8_t axis,
            double rot_vec[]);

    template<class T>
    void ignore_unused(T&) {}
};


class Similarity2D {
public:
    std::vector<Point2d> src;
    std::vector<Point2d> dst;

    std::vector<Point2d> src_transformed;

    /**
     * @brief t_x x-com
     */
    double t_x = 0;
    double t_y = 0;
    double angle = 0;
    double scale = 1;

    double fixed_scale = -1;

    /**
     * @brief outlier_threshold Maximum initial distance between source and destination point.
     */
    double outlier_threshold = -1;

    double cauchy_param = -1;

    double ceres_tolerance = 1e-12;

    /**
     * @brief max_movement Maximum movement (= max(length(src[i] - transform(src[i])))
     */
    double max_movement = -1;
    cv::Point2d max_movement_pt = {0,0};

    /**
     * @brief max_scale_movement Maximum movement caused by the scaling.
     */
    double max_scale_movement = -1;

    /**
     * @brief max_rotate_movement Maximum movement caused by the rotation.
     */
    double max_rotate_movement = -1;

    /**
     * @brief ignored Number of ignored correspondences due to outlier_threshold
     */
    size_t ignored = 0;

    /**
     * @brief t_length Length of the translation vector.
     */
    double t_length = 0;

    bool verbose = false;

    struct Similarity2DCost {
        cv::Point2d src, dst;
    public:
        Similarity2DCost(cv::Point2d const _src, cv::Point2d const _dst);

        template<class T>
        bool operator ()(
                T const * const angle,
                T const * const scale,
                T const * const t_x,
                T const * const t_y,
                T *residuals
                ) const;

        template<class T>
        static void transform(T const& angle, T const& scale, T const& t_x, T const& t_y, cv::Point2d const& src, T & dst_x, T& dst_y);
        static cv::Point2d transform(double const angle, double const scale, double const t_x, double const t_y,
                                     cv::Point2d const src);
    };

    cv::Point2d transform(cv::Point2d const src);


    Similarity2D(std::vector<cv::Point2d> const& _src = std::vector<cv::Point2d>(),
                 std::vector<cv::Point2d> const& _dst = std::vector<cv::Point2d>());

    /**
 * @brief fit2Dsimilarity Fit a 2d similarity transform mapping a given set of 2D-2D correspondences
 * dst ~= rotate (scale * src) + (x,y)
 */
    void runFit();

    void print(std::ostream &out) const;
};

struct FitGrid {
    std::vector<runningstats::QuantileStats<float> > per_grid_type_stats_x;
    std::vector<runningstats::QuantileStats<float> > per_grid_type_stats_y;
    std::vector<runningstats::QuantileStats<float> > per_grid_type_stats_z;
    std::vector<runningstats::QuantileStats<float> > per_grid_type_stats_length;

    double scale = 1;

    double ceres_tolerance = 1e-16;

    void findGrids(std::map<std::string, std::map<std::string, std::vector<cv::Point3f> > > &detected_grids,
                   const GridDescription &desc, Calib &calib, CalibResult & calib_result, std::vector<Point3f> initial_points);
    void findGridsSchilling(std::map<std::string, std::map<std::string, std::pair<std::vector<cv::Scalar_<int> >, std::vector<cv::Point3f> > > > &detected_grids, const GridDescription &desc);
public:
    string runFit(Calib &calib, CalibResult &calib_result, const std::vector<GridDescription> &desc);
    void runSchilling(const std::vector<GridDescription> &desc);
    void plotOffsetCorrectionSchilling(const std::vector<cv::Scalar_<int> > &_ids, const std::vector<cv::Point3f> &pts);
};

void write(cv::FileStorage& fs, const std::string&, const Calib& x);
void read(const cv::FileNode& node, Calib& x, const Calib& default_value = Calib());

std::string type2str(int type);

}

#endif // HDCALIB_H
