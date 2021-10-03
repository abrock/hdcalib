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

void CornerStore::addConditional(const std::vector<Corner> &vec) {
    for (auto const& pt : vec) {
        push_conditional(pt);
    }
}

void CornerStore::getMajorPoints(std::vector<Point2f> &imagePoints,
        std::vector<Point3f> &objectPoints,
        std::vector<cv::Scalar_<int> > &marker_references,
        hdcalib::Calib const& calib) const {
    imagePoints.clear();
    objectPoints.clear();
    marker_references.clear();
    for (hdmarker::Corner const& c : getMainMarkerAdjacent()) {
        if (calib.isValidPage(c)) {
            imagePoints.push_back(c.p);
            objectPoints.push_back(calib.getInitial3DCoord(c));
            marker_references.push_back(calib.getSimpleIdLayer(c));
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

size_t CornerStore::purgeUnlikelyByImageSize(const Size &size) {
    std::vector<Corner> replacement;
    replacement.reserve(corners.size());
    for (Corner const& it : corners) {
        if (it.p.x > 0 && it.p.y > 0 && it.p.x < size.width && it.p.y < size.height) {
            replacement.push_back(it);
        }
    }
    size_t const diff = corners.size() - replacement.size();
    if (diff > 0) {
        replaceCorners(replacement);
    }
    return diff;
}

void CornerStore::purgeRecursionDeeperThan(int level) {
    std::vector<Corner> newvec;
    for (Corner const& c : corners) {
        if (c.layer <= level) {
            newvec.push_back(c);
        }
    }
    if (newvec.size() < corners.size()) {
        replaceCorners(newvec);
    }
}

std::map<int, size_t> CornerStore::countLayers(std::vector<Corner> const& vec) {
    std::map<int, size_t> result;
    for (Corner const& c : vec) {
        if (c.layer >= 0) {
            result[c.layer]++;
        }
    }
    return result;
}

std::map<int, size_t> CornerStore::countLayers() const  {
    return countLayers(corners);
}

size_t CornerStore::countMainMarkers() const {
    return Calib::countMainMarkers(corners);
}

void CornerStore::sort() {
    std::vector<Corner> sorted = corners;
    std::sort(sorted.begin(), sorted.end());
    for (size_t ii = 0; ii < sorted.size(); ++ii) {
        if (corners[ii] != sorted[ii]) {
            replaceCorners(sorted);
            return;
        }
    }
}

size_t CornerStore::lastCleanDifference() const {
    return last_clean_diff;
}

int CornerStore::getCornerIdFactorFromMainMarkers(std::vector<Corner> const& vec) {
    int gcd = -1;
    for (Corner const& c : vec) {
        if (c.layer == 0) {
            gcd = Calib::tolerantGCD(gcd, c.id.x);
            gcd = Calib::tolerantGCD(gcd, c.id.y);
        }
    }
    return gcd;
}

int CornerStore::getCornerIdFactorFromMainMarkers() const {
    return getCornerIdFactorFromMainMarkers(corners);
}

std::vector<std::vector<Corner> > CornerStore::getSquares(int cornerIdFactor, runningstats::QuantileStats<float> *distances) const
{
    std::vector<std::vector<Corner> > result;
    cornerIdFactor = getCornerIdFactorFromMainMarkers();
    for (auto const& c : corners) {
        if (c.id.x % cornerIdFactor == 0 && c.id.y % cornerIdFactor == 0) {
            bool has_square = true;
            hdmarker::Corner neighbour;
            std::vector<hdmarker::Corner> square;
            square.push_back(c);
            for (cv::Point2i const & id_offset : {
                 cv::Point2i(0, cornerIdFactor),
                 cv::Point2i(cornerIdFactor, cornerIdFactor),
                 cv::Point2i(cornerIdFactor, 0)}) {
                hdmarker::Corner search = c;
                search.id += id_offset;
                if (hasID(search, neighbour)) {
                    square.push_back(neighbour);
                    if (distances) {
                        cv::Point2f const residual = c.p - neighbour.p;
                        double const dist = 2*double(std::sqrt(residual.dot(residual)));
                        distances->push_unsafe(dist/std::sqrt(id_offset.dot(id_offset)));
                    }
                }
                else {
                    has_square = false;
                }
            }
            if (has_square) {
                result.push_back(square);
            }
        }
    }

    return result;
}

std::vector<Corner> CornerStore::getSquaresTopLeft(int const cornerIdFactor, runningstats::QuantileStats<float> * distances) const {
    std::vector<hdmarker::Corner> result;
    for (auto const& it : getSquares(cornerIdFactor, distances)) {
        result.push_back(it[0]);
    }
    return result;
}

std::vector<hdmarker::Corner> CornerStore::getMainMarkers(const int cornerIdFactor) const {
    std::vector<hdmarker::Corner> result;
    for (hdmarker::Corner const& c : corners) {
        if (0 == c.layer ||
                (0 == (c.id.x % cornerIdFactor)
                && 0 == (c.id.y % cornerIdFactor))) {
            result.push_back(c);
        }
    }
    return result;
}

size_t CornerStore::distanceID(Corner const& a, Corner const& b) {
    return std::abs(a.id.x - b.id.x) + std::abs(a.id.y - b.id.y);
}

std::vector<Corner> CornerStore::getMainMarkerAdjacent(int const offset_x, int const offset_y) const {
    std::vector<Corner> const main_markers = getMainMarkers();

    std::vector<Corner> result;

    for (Corner const & main : main_markers) {
        cv::Point3i search(main.id.x + offset_x, main.id.y + offset_y, main.page);
        Corner found;
        if (hasID(search, found)) {
            result.push_back(found);
        }
    }

    return result;
}

std::vector<Corner> CornerStore::getMainMarkerAdjacent() const {
    std::vector<Corner> const main_markers = getMainMarkers();

    std::vector<Corner> result;

    for (Corner const & main : main_markers) {
        for (int offset : {1,3,5,15,25}){
            cv::Point3i search(main.id.x + offset, main.id.y + offset, main.page);
            Corner found;
            if (hasID(search, found)) {
                result.push_back(found);
                break;
            }
        }
    }

    return result;
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
    //purgeDuplicates();
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

void CornerStore::scaleIDs(int factor) {
    for (Corner& c : corners) {
        c.id *= factor;
    }
    replaceCorners(corners);
}

std::vector<Corner> CornerStore::getCorners() const {
    return corners;
}

std::vector<Corner> CornerStore::findByID(const Corner &ref, const size_t num_results) const {
    cv::Point3i const simpleID = Calib::getSimpleId(ref);
    return findByID(simpleID, num_results);
}

std::vector<Corner> CornerStore::findByID(const Point3i &ref, const size_t num_results) const {
    double query_pt[3] = {
        static_cast<double>(ref.x),
        static_cast<double>(ref.y),
        static_cast<double>(ref.z)
    };
    std::vector<hdmarker::Corner> result;

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

    last_clean_diff = size() - keep.size();

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

bool CornerStore::purgeOutOfBounds(const int min_x, const int min_y, const int max_x, const int max_y) {
    std::vector<hdmarker::Corner> keep;
    keep.reserve(size());
    for (hdmarker::Corner const& c : corners) {
        if (c.id.x >= min_x
                && c.id.y >= min_y
                && c.id.x <= max_x
                && c.id.y <= max_y) {
            keep.push_back(c);
        }
    }
    if (size() != keep.size()) {
        replaceCorners(keep);
        return true;
    }
    return false;
}

string Calib::printAllCameraMatrices() {
    std::stringstream result;
    for (auto & it : calibrations) {
        double const fx = it.second.cameraMatrix(0,0);
        double const fy = it.second.cameraMatrix(1,1);
        result << std::endl << it.first << std::endl
               << it.second.cameraMatrix << std::endl
               << "fx/fy: " << fx / fy << std::endl
               << "2|f_x-f_y|/(f_x+f_y): " << (2*std::abs(fx-fy)/(fx+fy)) << std::endl
               << std::endl;
    }
    return result.str();
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
    hdmarker::Corner c;
    return hasID(ref, c);
}

bool CornerStore::hasID(const Corner &ref, Corner &result) const {
    return hasID(Calib::getSimpleId(ref), result);
}

bool CornerStore::hasID(const Point3i &ref, Corner &found) const {
    double query_pt[3] = {
        static_cast<double>(ref.x),
        static_cast<double>(ref.y),
        static_cast<double>(ref.z)
    };
    size_t res_index;
    double res_dist_sqr;
    nanoflann::KNNResultSet<double> resultSet(1);
    resultSet.init(&res_index, &res_dist_sqr);
    idx_tree->findNeighbors(resultSet, query_pt, nanoflann::SearchParams(10));

    if (resultSet.size() < 1) {
        return false;
    }
    found = hdmarker::Corner(corners[res_index]);
    return found.id.x == ref.x && found.id.y == ref.y && found.page == ref.z;
}

bool CornerStore::hasIDLevel(const Corner &ref, Corner &found, int8_t level) const {
    cv::Scalar_<int> id = Calib::getSimpleIdLayer(ref);
    id[3] = level;
    return hasIDLevel(id, found);
}

bool CornerStore::hasIDLevel(const Corner &ref, int8_t level) const {
    hdmarker::Corner c;
    return hasIDLevel(ref, c, level);
}

bool CornerStore::hasIDLevel(cv::Scalar_<int> const& id, Corner &found) const
{
    double query_pt[3] = {
        static_cast<double>(id[0]),
        static_cast<double>(id[1]),
        static_cast<double>(id[2])
    };
    size_t res_index[4] = {0,0,0,0};
    double res_dist_sqr[4];
    nanoflann::KNNResultSet<double> resultSet(4);
    resultSet.init(res_index, res_dist_sqr);
    idx_tree->findNeighbors(resultSet, query_pt, nanoflann::SearchParams(10));

    if (resultSet.size() < 1) {
        return false;
    }
    for (size_t ii = 0; ii < resultSet.size(); ++ii) {
        found = hdmarker::Corner(corners[res_index[ii]]);
        if (found.id.x == id[0] && found.id.y == id[1] && found.page == id[2] && found.layer == id[3]) {
            return true;
        }
    }
    return false;
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

void Calib::keepCommonCorners_delete() {
    invalidateCache();
    CornerStore _delete;
    CornerStore _union = getUnion();

    for (size_t ii = 0; ii < _union.size(); ++ii) {
        hdmarker::Corner const c = _union.get(ii);
        for (auto const& it : data) {
            if (!(it.second.hasID(c))) {
                _delete.push_conditional(c);
                break;
            }
        }
    }

    for (auto& it : data) {
        bool found_delete = false;
        CornerStore replacement;
        for (size_t ii = 0; ii < it.second.size(); ++ii) {
            if (_delete.hasID(it.second.get(ii))) {
                found_delete = true;
            }
            else {
                replacement.push_conditional(it.second.get(ii));
            }
        }
        if (found_delete) {
            it.second = replacement;
        }
    }
}

void Calib::keepCommonCorners_intersect() {
    invalidateCache();

    if (data.empty() || data.size() < 2) {
        return;
    }
    std::string const last = std::prev(data.end())->first;
    if (data.size() == 2) {
        CornerStore::intersect(data.begin()->second, data[last]);
    }
    std::string prev = last;
    for (auto& it : data) {
        it.second.intersect(data[prev]);
        prev = it.first;
    }
    prev = std::prev(data.end())->first;
    for (auto& it : data) {
        it.second.intersect(data[prev]);
        prev = it.first;
    }
}

void Calib::keepCommonCorners() {
    invalidateCache();

    keepCommonCorners_intersect();
}


} // namespace hdcalib
