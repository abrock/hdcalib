#include "hdcalib.h"

namespace hdcalib {

size_t CornerStore::size() const {
    return corners.size();
}

const Corner &CornerStore::get(size_t index) const {
    if (index >= corners.size()) {
        throw std::out_of_range(std::string("Index ") + to_string(index) + " too large for current size of corners vector (" + to_string(corners.size()) + ")");
    }
    return corners[index];
}

void CornerStore::push_back(const Corner x) {
    corners.push_back(x);
    idx_tree->addPoints(corners.size()-1, corners.size()-1);
    pos_tree->addPoints(corners.size()-1, corners.size()-1);
}

void CornerStore::push_conditional(const Corner x) {
    if (!hasID(x)) {
        push_back(x);
    }
}

void CornerStore::add(const std::vector<Corner> &vec) {
    if (vec.empty()) {
        return;
    }
    corners.insert(corners.end(), vec.begin(), vec.end());
    idx_tree->addPoints(corners.size() - vec.size(), corners.size()-1);
    pos_tree->addPoints(corners.size() - vec.size(), corners.size()-1);
}

void CornerStore::getMajorPoints(
        std::vector<Point2f> &imagePoints,
        std::vector<Point3f> &objectPoints,
        hdcalib::Calib const& calib) const {
    imagePoints.clear();
    imagePoints.reserve(size());

    objectPoints.clear();
    objectPoints.reserve(size());
    for (size_t ii = 0; ii < size(); ++ii) {
        hdmarker::Corner const& c = get(ii);
        if (calib.isValidPage(c) && 0 == (c.id.x % 10) && 0 == (c.id.y % 10)) {
            imagePoints.push_back(get(ii).p);
            objectPoints.push_back(calib.getInitial3DCoord(get(ii)));
        }
    }
}

void CornerStore::getPoints(
        std::vector<Point2f> &imagePoints,
        std::vector<Point3f> &objectPoints,
        hdcalib::Calib const& calib) const {
    imagePoints.clear();
    imagePoints.reserve(size());

    objectPoints.clear();
    objectPoints.reserve(size());
    for (size_t ii = 0; ii < size(); ++ii) {
        if (calib.isValidPage(get(ii))) {
            imagePoints.push_back(get(ii).p);
            objectPoints.push_back(calib.getInitial3DCoord(get(ii)));
        }
    }
}

CornerStore::CornerStore() :
    idx_adapt(*this),
    pos_adapt(*this),
    idx_tree(new CornerIndexTree(
                 3 /*dim*/,
                 idx_adapt,
                 nanoflann::KDTreeSingleIndexAdaptorParams(16 /* max leaf */)
                 )),
    pos_tree(new CornerPositionTree (
                 2 /*dim*/,
                 pos_adapt,
                 nanoflann::KDTreeSingleIndexAdaptorParams(16 /* max leaf */)
                 )) {

}

CornerStore::CornerStore(const CornerStore &c) :
    idx_adapt(*this),
    pos_adapt(*this),
    idx_tree(new CornerIndexTree(
                 3 /*dim*/,
                 idx_adapt,
                 nanoflann::KDTreeSingleIndexAdaptorParams(16 /* max leaf */)
                 )),
    pos_tree(new CornerPositionTree (
                 2 /*dim*/,
                 pos_adapt,
                 nanoflann::KDTreeSingleIndexAdaptorParams(16 /* max leaf */)
                 )) {
    replaceCorners(c.getCorners());
}

CornerStore::CornerStore(const std::vector<Corner> &corners) :
    idx_adapt(*this),
    pos_adapt(*this),
    idx_tree(new CornerIndexTree(
                 3 /*dim*/,
                 idx_adapt,
                 nanoflann::KDTreeSingleIndexAdaptorParams(16 /* max leaf */)
                 )),
    pos_tree(new CornerPositionTree (
                 2 /*dim*/,
                 pos_adapt,
                 nanoflann::KDTreeSingleIndexAdaptorParams(16 /* max leaf */)
                 )) {
    replaceCorners(corners);
}

CornerStore &CornerStore::operator=(const CornerStore &other) {
    if (this != &other) { // protect against invalid self-assignment
        replaceCorners(other.getCorners());
    }
    // by convention, always return *this
    return *this;
}

void CornerStore::clean(int cornerIdFactor) {
    //purge32();
    purgeDuplicates();
    purgeUnlikely(cornerIdFactor);
}

void CornerStore::intersect(const CornerStore &b) {
    bool found_delete = false;
    std::vector<hdmarker::Corner> replacement;
    for (size_t ii = 0; ii < size(); ++ii) {
        if (b.hasID(get(ii))) {
            replacement.push_back(get(ii));
        }
        else {
            found_delete = true;
        }
    }
    if (found_delete) {
        replaceCorners(replacement);
    }
}

void CornerStore::intersect(CornerStore &a, CornerStore &b) {
    a.intersect(b);
    if (a.size() != b.size()) {
        b.intersect(a);
    }
}

void CornerStore::difference(const CornerStore &subtrahend) {
    bool found_delete = false;
    std::vector<hdmarker::Corner> replacement;
    for (size_t ii = 0; ii < size(); ++ii) {
        if (subtrahend.hasID(get(ii))) {
            found_delete = true;
        }
        else {
            replacement.push_back(get(ii));
        }
    }
    if (found_delete) {
        replaceCorners(replacement);
    }
}

void CornerStore::replaceCorners(const std::vector<Corner> &_corners) {
    corners = _corners;
    {
        std::shared_ptr<CornerIndexTree> idx_tree_replacement(new CornerIndexTree(
                                                                  3 /*dim*/,
                                                                  idx_adapt,
                                                                  nanoflann::KDTreeSingleIndexAdaptorParams(16 /* max leaf */)));
        idx_tree.swap(idx_tree_replacement);
    }
    {
        std::shared_ptr<CornerPositionTree> pos_tree_replacement(new CornerPositionTree(
                                                                     2 /*dim*/,
                                                                     pos_adapt,
                                                                     nanoflann::KDTreeSingleIndexAdaptorParams(16 /* max leaf */)));
        pos_tree.swap(pos_tree_replacement);
    }
    if (corners.size() > 0) {
        idx_tree->addPoints(0, corners.size() - 1);
        pos_tree->addPoints(0, corners.size() - 1);
    }
}

std::vector<Corner> CornerStore::getCorners() const {
    return corners;
}

std::vector<Corner> CornerStore::findByID(const Corner &ref, const size_t num_results) const {
    std::vector<hdmarker::Corner> result;
    double query_pt[3] = {
        static_cast<double>(ref.id.x),
        static_cast<double>(ref.id.y),
        static_cast<double>(ref.page)
    };

    // do a knn search
    std::unique_ptr<size_t[]> res_indices(new size_t[num_results]);
    std::unique_ptr<double[]> res_dist_sqr( new double[num_results]);
    nanoflann::KNNResultSet<double> resultSet(num_results);
    resultSet.init(res_indices.get(), res_dist_sqr.get());
    idx_tree->findNeighbors(resultSet, query_pt, nanoflann::SearchParams(10));

    for (size_t ii = 0; ii < resultSet.size(); ++ii) {
        result.push_back(corners[res_indices[ii]]);
    }

    return result;
}

std::vector<Corner> CornerStore::findByPos(const Corner &ref, const size_t num_results) {
    return findByPos(ref.p.x, ref.p.y, num_results);
}

std::vector<Corner> CornerStore::findByPos(const double x, const double y, const size_t num_results) {
    std::vector<hdmarker::Corner> result;
    double query_pt[2] = {
        static_cast<double>(x),
        static_cast<double>(y)
    };

    // do a knn search
    std::unique_ptr<size_t[]> res_indices(new size_t[num_results]);
    std::unique_ptr<double[]> res_dist_sqr( new double[num_results]);
    for (size_t ii = 0; ii < num_results; ++ii) {
        res_indices[ii] = 0;
        res_dist_sqr[ii] = std::numeric_limits<double>::max();
    }
    nanoflann::KNNResultSet<double> resultSet(num_results);
    resultSet.init(res_indices.get(), res_dist_sqr.get());
    pos_tree->findNeighbors(resultSet, query_pt, nanoflann::SearchParams(10));

    for (size_t ii = 0; ii < resultSet.size(); ++ii) {
        size_t const index = res_indices[ii];
        hdmarker::Corner const& c = corners[index];
        result.push_back(c);
    }

    return result;
}

bool CornerStore::purgeUnlikely(int cornerIdFactor) {
    std::vector<hdmarker::Corner> keep;
    keep.reserve(size());

    for (size_t ii = 0; ii < size(); ++ii) {
        hdmarker::Corner const& candidate = get(ii);
        std::vector<hdmarker::Corner> const res = findByPos(candidate, 5);
        size_t neighbours = 0;
        for (hdmarker::Corner const& neighbour : res) {
            if (neighbour.page != candidate.page) {
                continue;
            }
            if (neighbour.id == candidate.id) {
                continue;
            }
            cv::Point2i residual = candidate.id - neighbour.id;
            if (std::abs(residual.x) <= cornerIdFactor && std::abs(residual.y) <= cornerIdFactor) {
                neighbours++;
                if (neighbours > 1) {
                    keep.push_back(candidate);
                    break;
                }
            }
        }
    }

    if (size() != keep.size()) {
        replaceCorners(keep);
        return true;
    }
    return false;
}

bool CornerStore::purgeDuplicates() {
    std::vector<hdmarker::Corner> keep;
    keep.reserve(size());

    for (size_t ii = 0; ii < size(); ++ii) {
        hdmarker::Corner const& candidate = get(ii);
        size_t const num_results = 16;
        double query_pt[2] = {
            static_cast<double>(candidate.p.x),
            static_cast<double>(candidate.p.y)
        };

        // do a knn search
        std::unique_ptr<size_t[]> res_indices(new size_t[num_results]);
        std::unique_ptr<double[]> res_dist_sqr( new double[num_results]);
        nanoflann::KNNResultSet<double> resultSet(num_results);
        resultSet.init(res_indices.get(), res_dist_sqr.get());
        pos_tree->findNeighbors(resultSet, query_pt, nanoflann::SearchParams(10));

        bool is_duplicate = false;
        for (size_t jj = 0; jj < resultSet.size(); ++jj) {
            if (ii <= res_indices[jj]) {
                continue;
            }
            hdmarker::Corner const& b = get(res_indices[jj]);
            double const dist = Calib::distance(candidate.p, b.p);
            if (dist < (candidate.size + b.size)/20 && dist < 5) {
                is_duplicate = true;
                break;
            }
            break;
        }
        if (!is_duplicate) {
            keep.push_back(candidate);
        }
    }

    if (size() != keep.size()) {
        replaceCorners(keep);
        return true;
    }
    return false;
}

bool CornerStore::purge32() {
    std::vector<hdmarker::Corner> res;
    res.reserve(size());
    for (size_t ii = 0; ii < size(); ++ii) {
        hdmarker::Corner const& candidate = get(ii);
        if (candidate.id.x != 32 && candidate.id.y != 32) {
            res.push_back(candidate);
            continue;
        }
        auto const search_res = findByPos(candidate, 2);
        if (search_res.size() < 2) {
            res.push_back(candidate);
            continue;
        }
        hdmarker::Corner const& second = search_res[1];
        cv::Point2f const diff = candidate.p - second.p;
        double const dist = std::sqrt(diff.dot(diff));
        if (dist > (candidate.size + second.size)/20) {
            res.push_back(candidate);
        }
    }
    if (res.size() != size()) {
        replaceCorners(res);
        return true;
    }
    return false;
}

bool CornerStore::hasID(const Corner &ref) const {
    double query_pt[3] = {
        static_cast<double>(ref.id.x),
        static_cast<double>(ref.id.y),
        static_cast<double>(ref.page)
    };
    size_t res_index;
    double res_dist_sqr;
    nanoflann::KNNResultSet<double> resultSet(1);
    resultSet.init(&res_index, &res_dist_sqr);
    idx_tree->findNeighbors(resultSet, query_pt, nanoflann::SearchParams(10));

    if (resultSet.size() < 1) {
        return false;
    }
    hdmarker::Corner const& result = corners[res_index];
    return result.id == ref.id && result.page == ref.page;
}

CornerIndexAdaptor::CornerIndexAdaptor(CornerStore &ref) : store(&ref){

}

size_t CornerIndexAdaptor::kdtree_get_point_count() const {
    return store->size();
}

int CornerIndexAdaptor::kdtree_get_pt(const size_t idx, int dim) const {
    hdmarker::Corner const& c = store->get(idx);
    if (0 == dim) {
        return c.id.x;
    }
    if (1 == dim) {
        return c.id.y;
    }
    if (2 == dim) {
        return c.page;
    }
    throw std::out_of_range("Dimension number " + to_string(dim) + " out of range (0-2)");
}

CornerPositionAdaptor::CornerPositionAdaptor(CornerStore &ref) : store(&ref) {

}

size_t CornerPositionAdaptor::kdtree_get_point_count() const {
    return store->size();
}

int CornerPositionAdaptor::kdtree_get_pt(const size_t idx, int dim) const {
    hdmarker::Corner const& c = store->get(idx);
    if (0 == dim) {
        return c.p.x;
    }
    if (1 == dim) {
        return c.p.y;
    }
    throw std::out_of_range(std::string("Requested dimension ") + to_string(dim) + " out of range (0-1)");
}

} // namespace hdcalib
