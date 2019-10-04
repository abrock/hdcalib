#include <iostream>

#include <gtest/gtest.h>

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <algorithm>
#include <random>

#include "hdmarker.hpp"
#include "hdcalib.h"

static std::random_device rd;
static std::default_random_engine engine(rd());
static std::normal_distribution<double> dist;

::testing::AssertionResult RelativeNear(const double a, const double b, double delta) {
    double const diff = std::abs(a-b);
    double const relative_diff = 2*diff/(std::abs(a) + std::abs(b));
    if (relative_diff < delta)
        return ::testing::AssertionSuccess();
    else
        return ::testing::AssertionFailure() << "The absolute difference is " << diff
                                             << ", the relative difference is " << relative_diff
                                             << " which exceeds " << delta;
}

::testing::AssertionResult rot4_eq(const double a[4], const double b[4],
const double threshold = 1e-12) {
    if (std::abs(a[3]) < 100*std::numeric_limits<double>::min()
            && std::abs(a[3]) < 100 * std::numeric_limits<double>::min())
        return ::testing::AssertionSuccess();
    for (size_t ii = 0; ii < 3; ++ii) {
        if (std::abs(a[ii] - b[ii]) > threshold) {
            return ::testing::AssertionFailure()
                    << "The difference between a[" << ii
                    << "] and b[" << ii << "] is " << std::abs(a[ii] - b[ii])
                    << ", which exceeds " << threshold << ", where a["
                    << ii << "] evaluates to " << a[ii]
                    << " and b[" << ii << "] evaluates to " << b[ii];
        }
    }
    return ::testing::AssertionSuccess();
}

void getCornerGrid(
        hdcalib::CornerStore & store,
        size_t const grid_width = 50,
        size_t const grid_height = 50,
        int const page = 0,
        cv::Point2f offset = cv::Point2f(0,0)) {
    hdmarker::Corner a;
    for (size_t ii = 0; ii < grid_width; ++ii) {
        for (size_t jj = 0; jj < grid_height; ++jj) {
            a.id = cv::Point2i(int(ii), int(jj));
            a.page = page;
            a.p = cv::Point2f(ii, jj) + offset;
            a.size = .5;
            store.push_back(a);
        }
    }
}

void getCornerGridConditional(
        hdcalib::CornerStore & store,
        size_t const grid_width = 50,
        size_t const grid_height = 50,
        int const page = 0) {
    hdmarker::Corner a;
    for (size_t ii = 0; ii < grid_width; ++ii) {
        for (size_t jj = 0; jj < grid_height; ++jj) {
            a.id = cv::Point2i(int(ii), int(jj));
            a.page = page;
            a.p = cv::Point2f(ii, jj);
            a.size = .5;
            store.push_conditional(a);
        }
    }
}

cv::Point2d randomPoint() {
    return cv::Point2d(dist(engine), dist(engine));
}

bool float_eq(float const a, float const b) {
    if (std::isnan(a) || std::isnan(b) || std::isinf(a) || std::isinf(b)) {
        return false;
    }
    if (std::abs(a) < std::numeric_limits<float>::epsilon() || std::abs(b) < std::numeric_limits<float>::epsilon()) {
        if (std::abs(a-b) < std::numeric_limits<float>::epsilon()) {
            return true;
        }
        else {
            return false;
        }
    }
    if (std::abs(a-b) / (std::abs(a) + std::abs(b)) < std::numeric_limits<float>::epsilon()) {
        return true;
    }
    return false;
}

bool point2f_eq(cv::Point2f const& a, cv::Point2f const& b) {
    return float_eq(a.x, b.x) && float_eq(a.y, b.y);
}

bool point2i_eq(cv::Point2i const& a, cv::Point2i const& b) {
    return a.x == b.x && a.y == b.y;
}

::testing::AssertionResult CornersEqual(hdmarker::Corner const& a, hdmarker::Corner const& b) {
    if (!point2f_eq(a.p, b.p)) {
        return ::testing::AssertionFailure() << "at p: " << a.p << " not equal to " << b.p;
    }
    for (size_t ii = 0; ii < 3; ++ii) {
        if (!point2f_eq(a.pc[ii], b.pc[ii])) {
            return ::testing::AssertionFailure() << "at pc[" << ii << "]: " << a.pc[ii] << " not equal to " << b.pc[ii];
        }
    }
    if (!point2i_eq(a.id, b.id)) {
        return ::testing::AssertionFailure() << "at id: " << a.id << " not equal to " << b.id;
    }
    if (a.page != b.page) {
        return ::testing::AssertionFailure() << "at page: " << a.page << " not equal to " << b.page;
    }
    if (!float_eq(a.size, b.size)) {
        return ::testing::AssertionFailure() << "at size: " << a.size << "not equal to " << b.size;
    }
    return ::testing::AssertionSuccess();
}

TEST(CornerStore, Adaptors) {
    hdcalib::CornerStore store;
    hdcalib::CornerIndexAdaptor idx_adapt(store);
    hdcalib::CornerPositionAdaptor pos_adapt(store);

    // We create a hdmarker::Corner with a different value for each property.
    hdmarker::Corner a;
    a.p = cv::Point2f(1,2);
    a.id = cv::Point2i(3,4);
    a.pc[0] = cv::Point2f(5,6);
    a.pc[1] = cv::Point2f(7,8);
    a.pc[2] = cv::Point2f(9,10);
    a.page = 11;
    a.size = 12;

    // We didn't put the Corner in the CornerStore yet so this should fail.
    EXPECT_THROW({store.get(0);}, std::out_of_range);

    store.push_back(a);

    hdmarker::Corner const& a_copy = store.get(0);

    EXPECT_THROW({store.get(1);}, std::out_of_range);

    EXPECT_TRUE(CornersEqual(a, a_copy));

    EXPECT_EQ(a.id.x, idx_adapt.kdtree_get_pt(0, 0));
    EXPECT_EQ(a.id.y, idx_adapt.kdtree_get_pt(0, 1));
    EXPECT_EQ(a.page, idx_adapt.kdtree_get_pt(0, 2));

    EXPECT_EQ(a.p.x, pos_adapt.kdtree_get_pt(0, 0));
    EXPECT_EQ(a.p.y, pos_adapt.kdtree_get_pt(0, 1));

    // Second Corner with different values than the first.
    hdmarker::Corner b;
    b.p = cv::Point2f(13,14);
    b.id = cv::Point2i(15,16);
    b.pc[0] = cv::Point2f(17,18);
    b.pc[1] = cv::Point2f(19,20);
    b.pc[2] = cv::Point2f(21,22);
    b.page = 23;
    b.size = 24;

    store.push_back(b);
    EXPECT_THROW({store.get(2);}, std::out_of_range);

    hdmarker::Corner const& b_copy = store.get(1);

    EXPECT_TRUE(CornersEqual(b, b_copy));

    EXPECT_EQ(b.id.x, idx_adapt.kdtree_get_pt(1, 0));
    EXPECT_EQ(b.id.y, idx_adapt.kdtree_get_pt(1, 1));
    EXPECT_EQ(b.page, idx_adapt.kdtree_get_pt(1, 2));

    EXPECT_EQ(b.p.x, pos_adapt.kdtree_get_pt(1, 0));
    EXPECT_EQ(b.p.y, pos_adapt.kdtree_get_pt(1, 1));
    EXPECT_NE(a.p.x, pos_adapt.kdtree_get_pt(1, 0));
    EXPECT_NE(a.p.y, pos_adapt.kdtree_get_pt(1, 1));

    EXPECT_THROW({idx_adapt.kdtree_get_pt(0, 3);}, std::out_of_range);
    EXPECT_THROW({idx_adapt.kdtree_get_pt(1, 3);}, std::out_of_range);
    EXPECT_THROW({idx_adapt.kdtree_get_pt(2, 0);}, std::out_of_range);
    EXPECT_THROW({idx_adapt.kdtree_get_pt(2, 1);}, std::out_of_range);
    EXPECT_THROW({idx_adapt.kdtree_get_pt(2, 2);}, std::out_of_range);
    EXPECT_THROW({idx_adapt.kdtree_get_pt(2, 3);}, std::out_of_range);

    size_t const num_markers = 10;
    for (size_t ii = store.size(); ii < num_markers; ++ii) {
        hdmarker::Corner x;
        x.p = cv::Point2f(ii*12+1,ii*12+2);
        x.id = cv::Point2i(int(ii*12+3),int(ii*12+4));
        x.pc[0] = cv::Point2f(ii*12+5,ii*12+6);
        x.pc[1] = cv::Point2f(ii*12+7,ii*12+8);
        x.pc[2] = cv::Point2f(ii*12+9,ii*12+10);
        x.page = int(ii*12+11);
        x.size = ii*12+12;

        store.push_back(x);
        EXPECT_THROW({store.get(store.size());}, std::out_of_range);

        hdmarker::Corner const& x_copy = store.get(ii);

        EXPECT_TRUE(CornersEqual(x, x_copy));

        EXPECT_EQ(x.id.x, idx_adapt.kdtree_get_pt(ii, 0));
        EXPECT_EQ(x.id.y, idx_adapt.kdtree_get_pt(ii, 1));
        EXPECT_EQ(x.page, idx_adapt.kdtree_get_pt(ii, 2));

        EXPECT_EQ(x.p.x, pos_adapt.kdtree_get_pt(ii, 0));
        EXPECT_EQ(x.p.y, pos_adapt.kdtree_get_pt(ii, 1));

        EXPECT_THROW({idx_adapt.kdtree_get_pt(0, 3);}, std::out_of_range);
        EXPECT_THROW({idx_adapt.kdtree_get_pt(1, 3);}, std::out_of_range);
        EXPECT_THROW({idx_adapt.kdtree_get_pt(ii+1, 0);}, std::out_of_range);
        EXPECT_THROW({idx_adapt.kdtree_get_pt(ii+1, 1);}, std::out_of_range);
        EXPECT_THROW({idx_adapt.kdtree_get_pt(ii+1, 2);}, std::out_of_range);
        EXPECT_THROW({idx_adapt.kdtree_get_pt(ii+1, 3);}, std::out_of_range);
    }

}


TEST(CornerStore, findByID) {
    hdcalib::CornerStore store;
    hdcalib::CornerIndexAdaptor idx_adapt(store);
    hdcalib::CornerPositionAdaptor pos_adapt(store);


    // We create a hdmarker::Corner with a different value for each property.
    hdmarker::Corner a, b;
    a.p = cv::Point2f(1,2);
    a.id = cv::Point2i(3,4);
    a.pc[0] = cv::Point2f(5,6);
    a.pc[1] = cv::Point2f(7,8);
    a.pc[2] = cv::Point2f(9,10);
    a.page = 11;
    a.size = 12;

    EXPECT_FALSE(store.hasID(a));
    EXPECT_FALSE(store.hasID(b));

    store.push_back(a);
    EXPECT_TRUE(store.hasID(a));
    EXPECT_FALSE(store.hasID(b));

    b.p = cv::Point2f(13,14);
    b.id = cv::Point2i(15,16);
    b.pc[0] = cv::Point2f(17,18);
    b.pc[1] = cv::Point2f(19,20);
    b.pc[2] = cv::Point2f(21,22);
    b.page = 23;
    b.size = 24;

    std::vector<hdmarker::Corner> search_res = store.findByID(a);
    search_res = store.findByID(b);
    ASSERT_GE(search_res.size(), 1);
    EXPECT_TRUE(CornersEqual(a, search_res[0]));

    store.push_back(b);
    EXPECT_TRUE(store.hasID(a));
    EXPECT_TRUE(store.hasID(b));


    search_res = store.findByID(a);
    ASSERT_GE(search_res.size(), 1);
    EXPECT_TRUE(CornersEqual(a, search_res[0]));

    store.push_back(b);
    EXPECT_TRUE(store.hasID(a));
    EXPECT_TRUE(store.hasID(b));

    search_res = store.findByID(a);
    ASSERT_GE(search_res.size(), 1);
    EXPECT_TRUE(CornersEqual(a, search_res[0]));

    store.push_back(b);
    EXPECT_TRUE(store.hasID(a));
    EXPECT_TRUE(store.hasID(b));

    search_res = store.findByID(a);
    ASSERT_GE(search_res.size(), 1);
    EXPECT_TRUE(CornersEqual(a, search_res[0]));

}

TEST(CornerStore, find) {
    hdcalib::CornerStore store;
    hdmarker::Corner a;

    size_t const grid_width = 50;
    size_t const grid_height = 50;

    getCornerGrid(store, grid_width, grid_height);
    for (size_t ii = 0; ii < grid_width; ++ii) {
        for (size_t jj = 0; jj < grid_height; ++jj) {
            a.id = cv::Point2i(int(ii), int(jj));
            a.page = 0;
            a.p = cv::Point2f(ii, jj);
            a.size = .5;
            std::vector<hdmarker::Corner> res = store.findByPos(a, 1);
            ASSERT_EQ(1, res.size());
            EXPECT_TRUE(CornersEqual(a, res[0]));

            store.findByID(a, 1);
            ASSERT_EQ(1, res.size());
            EXPECT_TRUE(CornersEqual(a, res[0]));
            EXPECT_TRUE(store.hasID(a));
        }
    }

    std::uniform_real_distribution<float> dist_x(-.4f, float(grid_width) - .6f);
    std::uniform_real_distribution<float> dist_y(-.4f, float(grid_height) - .6f);
    for (size_t ii = 0; ii < 10*1000; ++ii) {
        float const x = dist_x(engine);
        float const y = dist_y(engine);
        cv::Point2i id(int(std::round(x)), int(std::round(y)));
        std::vector<hdmarker::Corner> res = store.findByPos(double(x), double(y), 1);
        ASSERT_EQ(1, res.size());
        hdmarker::Corner const& r = res[0];
        EXPECT_NEAR(x, r.p.x, .50001);
        EXPECT_NEAR(y, r.p.y, .50001);
        EXPECT_EQ(id.x, r.id.x);
        EXPECT_EQ(id.y, r.id.y);
    }

    size_t const old_size = store.size();
    { // Insert an unlikely Corner into the database
        a.page = 150;
        a.id.x = 32;
        store.push_back(a);
    }
    store.purgeUnlikely(1);
    EXPECT_EQ(store.size(), old_size);

    for (size_t ii = 0; ii < grid_width; ++ii) {
        for (size_t jj = 0; jj < grid_height; ++jj) {
            a.id = cv::Point2i(int(ii), int(jj));
            a.page = 0;
            a.p = cv::Point2f(ii, jj);
            a.size = .5;
            std::vector<hdmarker::Corner> res = store.findByPos(a, 1);
            ASSERT_EQ(1, res.size());
            EXPECT_TRUE(CornersEqual(a, res[0]));

            store.findByID(a, 1);
            ASSERT_EQ(1, res.size());
            EXPECT_TRUE(CornersEqual(a, res[0]));
            EXPECT_TRUE(store.hasID(a));
        }
    }

    for (size_t ii = 0; ii < 10*1000; ++ii) {
        float const x = dist_x(engine);
        float const y = dist_y(engine);
        cv::Point2i id(int(std::round(x)), int(std::round(y)));
        std::vector<hdmarker::Corner> res = store.findByPos(double(x), double(y), 1);
        ASSERT_EQ(1, res.size());
        hdmarker::Corner const& r = res[0];
        EXPECT_NEAR(x, r.p.x, .50001);
        EXPECT_NEAR(y, r.p.y, .50001);
        EXPECT_EQ(id.x, r.id.x);
        EXPECT_EQ(id.y, r.id.y);
    }
}

TEST(CornerStore, purgeDuplicates) {
    hdcalib::CornerStore store;
    hdmarker::Corner a;

    size_t const grid_width = 50;
    size_t const grid_height = 50;

    getCornerGrid(store, grid_width, grid_height);
    size_t const grid_size = store.size();

    EXPECT_FALSE(store.purgeDuplicates());
    EXPECT_EQ(store.size(), grid_size);

    getCornerGrid(store, 5, 5);

    EXPECT_TRUE(store.purgeDuplicates());
    EXPECT_EQ(store.size(), grid_size);

    getCornerGrid(store);

    EXPECT_TRUE(store.purgeDuplicates());
    EXPECT_EQ(store.size(), grid_size);

    hdcalib::CornerStore store2;
    getCornerGrid(store2, grid_width, grid_height);

    EXPECT_EQ(store.size(), store2.size());

    for (size_t ii = 0; ii < store2.size(); ++ii) {
        EXPECT_TRUE(store.hasID(store2.get(ii)));
    }

    hdcalib::CornerStore store3;
    store3 = store;
}

TEST(CornerStore, copyConstructor) {
    hdcalib::CornerStore store;
    hdmarker::Corner a, b;

    a.p = cv::Point2f(1,2);
    a.id = cv::Point2i(3,4);
    a.pc[0] = cv::Point2f(5,6);
    a.pc[1] = cv::Point2f(7,8);
    a.pc[2] = cv::Point2f(9,10);
    a.page = 11;
    a.size = 12;

    b.p = cv::Point2f(1,2);
    b.id = cv::Point2i(3,4);
    b.pc[0] = cv::Point2f(5,6);
    b.pc[1] = cv::Point2f(7,8);
    b.pc[2] = cv::Point2f(9,10);
    b.page = 42;
    b.size = 12;

    size_t const grid_width = 50;
    size_t const grid_height = 50;

    getCornerGrid(store, grid_width, grid_height);
    //size_t const grid_size = store.size();

    hdcalib::CornerStore copy = store;

    EXPECT_EQ(copy.size(), store.size());
    EXPECT_EQ(copy.size(), 2500);
    EXPECT_EQ(store.size(), 2500);

    EXPECT_FALSE(store.hasID(a));
    EXPECT_FALSE(copy.hasID(a));
    EXPECT_FALSE(store.hasID(b));
    EXPECT_FALSE(copy.hasID(b));

    store.push_back(a);

    EXPECT_EQ(copy.size(), 2500);
    EXPECT_EQ(store.size(), 2501);

    EXPECT_TRUE(store.hasID(a));
    EXPECT_FALSE(copy.hasID(a));
    EXPECT_FALSE(store.hasID(b));
    EXPECT_FALSE(copy.hasID(b));

    copy.push_back(b);

    EXPECT_EQ(copy.size(), 2501);
    EXPECT_EQ(store.size(), 2501);

    EXPECT_TRUE(store.hasID(a));
    EXPECT_FALSE(copy.hasID(a));
    EXPECT_FALSE(store.hasID(b));
    EXPECT_TRUE(copy.hasID(b));

    std::map<std::string, hdcalib::CornerStore> map;
    map["store"] = store;
    map["copy"] = copy;

    EXPECT_EQ(map["store"].size(), 2501);
    EXPECT_EQ(map["copy"].size(), 2501);

    map["store"].intersect(map["copy"]);
    EXPECT_EQ(map["store"].size(), 2500);
    EXPECT_EQ(map["copy"].size(), 2501);

    map["copy"].intersect(map["store"]);
    EXPECT_EQ(map["store"].size(), 2500);
    EXPECT_EQ(map["copy"].size(), 2500);

    map["copy"].intersect(map["copy"]);
    map["store"].intersect(map["store"]);
    EXPECT_EQ(map["store"].size(), 2500);
    EXPECT_EQ(map["copy"].size(), 2500);

    {
        hdcalib::Calib c;
        c.addInputImage("store", map["store"]);
        c.addInputImage("copy", map["copy"]);

        EXPECT_EQ(c.get("store").size(), 2500);
        EXPECT_EQ(c.get("copy").size(), 2500);



        c.keepCommonCorners_intersect();
        EXPECT_EQ(c.get("store").size(), 2500);
        EXPECT_EQ(c.get("copy").size(), 2500);
    }
}

TEST(CornerStore, purge) {
    hdcalib::CornerStore s;
    getCornerGrid(s, 30, 30, 0);
    EXPECT_EQ(s.size(), 900);

    s.purgeUnlikely(1);
    EXPECT_EQ(s.size(), 900);

    s.purgeDuplicates();
    EXPECT_EQ(s.size(), 900);

    getCornerGrid(s, 10, 10, 1, cv::Point2f(32,0));
    EXPECT_EQ(s.size(), 1000);

    s.purgeUnlikely(1);
    EXPECT_EQ(s.size(), 1000);

    s.purgeDuplicates();
    EXPECT_EQ(s.size(), 1000);
}

TEST(CornerStore, push_conditional) {
    hdcalib::CornerStore c;

    EXPECT_EQ(0, c.size());

    getCornerGridConditional(c, 20, 20, 0);
    EXPECT_EQ(400, c.size());

    getCornerGridConditional(c, 20, 20, 0);
    EXPECT_EQ(400, c.size());

    getCornerGridConditional(c, 30, 30, 0);
    EXPECT_EQ(900, c.size());

    getCornerGridConditional(c, 10, 10, 1);
    EXPECT_EQ(1000, c.size());

    getCornerGridConditional(c, 20, 20, 1);
    EXPECT_EQ(1300, c.size());
}

TEST(Calib, keepCommonCorners_delete) {
    std::vector<std::string> names = {"a.png", "b.png"};
    hdcalib::Calib c;
    for (auto& name : names) {
        hdcalib::CornerStore s;
        getCornerGrid(s, 50, 50, 0);
        c.addInputImage(name, s);
        EXPECT_EQ(c.get(name).size(), 2500);
    }

    hdcalib::CornerStore _union = c.getUnion();

    EXPECT_EQ(_union.size(), 2500);

    for (auto const& name : names) {
        EXPECT_EQ(c.get(name).size(), 2500);
    }

    c.keepCommonCorners_intersect();

    for (const auto& name : names) {
        EXPECT_EQ(c.get(name).size(), 2500);
    }
}

TEST(CornerStore, intersect) {
    hdcalib::CornerStore a,b;

    getCornerGrid(a, 50, 50, 0);
    getCornerGrid(b, 50, 50, 0);

    EXPECT_EQ(a.size(), 2500);
    EXPECT_EQ(b.size(), 2500);

    a.intersect(b);
    EXPECT_EQ(a.size(), 2500);
    EXPECT_EQ(b.size(), 2500);

    a.intersect(a);
    EXPECT_EQ(a.size(), 2500);
    EXPECT_EQ(b.size(), 2500);

    b.intersect(a);
    EXPECT_EQ(a.size(), 2500);
    EXPECT_EQ(b.size(), 2500);

    getCornerGrid(a, 10, 10, 1);
    EXPECT_EQ(a.size(), 2600);

    a.intersect(b);
    EXPECT_EQ(a.size(), 2500);
    EXPECT_EQ(b.size(), 2500);
}

TEST(CornerStore, purge2) {
    hdcalib::CornerStore a;

    getCornerGrid(a, 50, 50, 0);
    EXPECT_EQ(a.size(), 2500);

    getCornerGrid(a, 50, 50, 0);
    EXPECT_EQ(a.size(), 5000);

    a.purgeUnlikely(1);
    EXPECT_EQ(a.size(), 5000);

    a.purgeDuplicates();
    EXPECT_EQ(a.size(), 2500);

    getCornerGrid(a, 10, 10, 1, cv::Point2f(50, 50));
    EXPECT_EQ(a.size(), 2600);

    a.purgeUnlikely(1);
    EXPECT_EQ(a.size(), 2600);

    a.purgeDuplicates();
    EXPECT_EQ(a.size(), 2600);
}

TEST(CornerStore, assignment) {
    hdcalib:: CornerStore a, b;

    getCornerGrid(a, 10, 10, 0);
    getCornerGrid(b, 20, 20, 1);

    a=b;

    EXPECT_EQ(a.size(), 400);
    EXPECT_EQ(b.size(), 400);

    getCornerGrid(a, 30, 30, 0);
    getCornerGrid(b, 15, 15, 1);

    EXPECT_EQ(a.size(), 400+900);
    EXPECT_EQ(b.size(), 400+225);

}

class KeepCornersDelete {
public:
    static void keepCorners(hdcalib::Calib & c) {
        c.keepCommonCorners_delete();
    }
};

class KeepCornersIntersect {
public:
    static void keepCorners(hdcalib::Calib & c) {
        c.keepCommonCorners_intersect();
    }
};

template <typename T>
class KeepCornersTest : public ::testing::Test {};
using MyTypes = ::testing::Types<
KeepCornersDelete,
KeepCornersIntersect
>;

TYPED_TEST_CASE(KeepCornersTest, MyTypes);

TYPED_TEST(KeepCornersTest, empty_intersection) {
    hdcalib::CornerStore a, b;

    getCornerGrid(a, 30, 30, 0);
    getCornerGrid(b, 40, 40, 1);
    hdcalib::Calib c;
    c.addInputImage("a", a);
    c.addInputImage("b", b);

    EXPECT_EQ(c.get("a").size(), 900);
    EXPECT_EQ(c.get("b").size(), 1600);

    TypeParam::keepCorners(c);
    EXPECT_EQ(c.get("a").size(), 0);
    EXPECT_EQ(c.get("b").size(), 0);
}


TYPED_TEST(KeepCornersTest, identical) {
    hdcalib::CornerStore a, b;

    getCornerGrid(a, 30, 30, 0);
    getCornerGrid(b, 30, 30, 0);
    hdcalib::Calib c;
    c.addInputImage("a", a);
    c.addInputImage("b", b);

    EXPECT_EQ(c.get("a").size(), 900);
    EXPECT_EQ(c.get("b").size(), 900);

    TypeParam::keepCorners(c);
    EXPECT_EQ(c.get("a").size(), 900);
    EXPECT_EQ(c.get("b").size(), 900);
}

TYPED_TEST(KeepCornersTest, non_identical) {
    hdcalib::CornerStore a, b;

    getCornerGrid(a, 30, 30, 0);
    getCornerGrid(b, 30, 30, 0);

    getCornerGrid(a, 10, 10, 1, cv::Point2f(30, 30));
    getCornerGrid(b, 5, 5, 1, cv::Point2f(30, 30));
    hdcalib::Calib c;
    c.addInputImage("a", a);
    c.addInputImage("b", b);

    EXPECT_EQ(c.get("a").size(), 1000);
    EXPECT_EQ(c.get("b").size(), 925);

    TypeParam::keepCorners(c);
    EXPECT_EQ(c.get("a").size(), 925);
    EXPECT_EQ(c.get("b").size(), 925);
}

/**
 * @brief getNames generates an array of different strings.
 * @param num number of strings we want.
 * @return
 */
std::vector<std::string> getNames(size_t const num) {
    std::vector<std::string> res;
    for (size_t ii = 0; ii < num; ++ii) {
        res.push_back(std::to_string(ii));
    }
    return res;
}

TYPED_TEST(KeepCornersTest, identical_different_sizes) {
    for (size_t num = 3; num < 10; ++num) {
        auto const names = getNames(num);
        hdcalib::Calib c;
        for (auto const& name : names) {
            hdcalib::CornerStore s;
            getCornerGrid(s, 5, 5, 0);
            c.addInputImage(name, s);
        }
        for (auto const& name : names) {
            EXPECT_EQ(c.get(name).size(), 25);
        }
        TypeParam::keepCorners(c);
        for (auto const& name : names) {
            EXPECT_EQ(c.get(name).size(), 25);
        }
    }
}

TYPED_TEST(KeepCornersTest, empty_intersection_different_sizes) {
    for (size_t num = 3; num < 10; ++num) {
        auto const names = getNames(num);
        hdcalib::Calib c;
        int page = 0;
        for (auto const& name : names) {
            hdcalib::CornerStore s;
            getCornerGrid(s, 5, 5, page);
            c.addInputImage(name, s);
            page++;
        }
        for (auto const& name : names) {
            EXPECT_EQ(c.get(name).size(), 25);
        }
        TypeParam::keepCorners(c);
        for (auto const& name : names) {
            EXPECT_EQ(c.get(name).size(), 0);
        }
    }
}

TYPED_TEST(KeepCornersTest, non_empty_intersection_different_sizes) {
    for (size_t num = 1; num < 10; ++num) {
        auto const names = getNames(num);
        hdcalib::Calib c;
        size_t extension_size = 3;
        for (auto const& name : names) {
            hdcalib::CornerStore s;
            getCornerGrid(s, 5, 5, 0);
            getCornerGrid(s, 3, extension_size, 1, cv::Point2f(6,0));
            c.addInputImage(name, s);
            extension_size++;
        }
        extension_size = 3;
        for (auto const& name : names) {
            EXPECT_EQ(c.get(name).size(), 25 + extension_size*3);
            extension_size++;
        }
        TypeParam::keepCorners(c);
        for (auto const& name : names) {
            EXPECT_EQ(c.get(name).size(), 25 + 3*3);
        }
    }
}

TEST(CornerStore, purge32) {
    {
        hdcalib::CornerStore s;
        getCornerGrid(s, 15, 15);
        EXPECT_EQ(s.size(), 225);
        EXPECT_FALSE(s.purge32());
        EXPECT_EQ(s.size(), 225);
    }
    {
        hdcalib::CornerStore s;
        getCornerGrid(s, 40, 40);
        EXPECT_EQ(s.size(), 1600);
        EXPECT_FALSE(s.purge32());
        EXPECT_EQ(s.size(), 1600);
    }
    {
        hdcalib::CornerStore s;
        getCornerGrid(s, 50, 50);
        EXPECT_EQ(s.size(), 2500);
        EXPECT_FALSE(s.purge32());
        EXPECT_EQ(s.size(), 2500);
    }
    {
        hdcalib::CornerStore s;
        getCornerGrid(s, 33, 33);
        EXPECT_EQ(s.size(), 33*33);
        EXPECT_FALSE(s.purge32());
        EXPECT_EQ(s.size(), 33*33);
        getCornerGrid(s, 3, 32, 1, cv::Point2f(32, 0));
        EXPECT_EQ(s.size(), 33*33 + 3*32);
        EXPECT_TRUE(s.purge32());
        EXPECT_EQ(s.size(), 33*33 + 2*32);

    }

}

TEST(CornerStore, difference) {
    {
        hdcalib::CornerStore a, b;
        getCornerGrid(a, 15, 15);
        EXPECT_EQ(a.size(), 225);

        getCornerGrid(b, 15, 15);
        EXPECT_EQ(b.size(), 225);

        a.difference(b);
        EXPECT_EQ(a.size(), 0);
        EXPECT_EQ(b.size(), 225);
    }
    {
        hdcalib::CornerStore a, b;
        getCornerGrid(a, 15, 15);
        EXPECT_EQ(a.size(), 225);

        getCornerGrid(b, 5, 5);
        EXPECT_EQ(b.size(), 25);

        a.difference(b);
        EXPECT_EQ(a.size(), 200);
        EXPECT_EQ(b.size(), 25);
    }
    {
        hdcalib::CornerStore a, b;
        getCornerGrid(a, 5, 5);
        EXPECT_EQ(a.size(), 25);

        getCornerGrid(b, 15, 15);
        EXPECT_EQ(b.size(), 225);

        a.difference(b);
        EXPECT_EQ(a.size(), 0);
        EXPECT_EQ(b.size(), 225);
    }
    {
        hdcalib::CornerStore a, b;
        getCornerGrid(a, 5, 5);
        EXPECT_EQ(a.size(), 25);

        EXPECT_EQ(b.size(), 0);

        a.difference(b);
        EXPECT_EQ(a.size(), 25);
        EXPECT_EQ(b.size(), 0);
    }
}
double square_p1(double const in) {
    return in*in+1;
}

TEST(CalibrationResult, project_simple) {
    for (size_t ii = 0; ii < 3000; ++ii) {
        double const X = dist(engine), Y = dist(engine), Z = square_p1(dist(engine)),
                f_x = square_p1(dist(engine)), f_y = square_p1(dist(engine)),
                p_x = dist(engine), p_y = dist(engine),
                t_x = dist(engine), t_y = dist(engine), t_z = square_p1(dist(engine));

        cv::Mat_<double> camera_matrix = {f_x,0,p_x,   0,f_y,p_y,   0,0,1};
        camera_matrix = camera_matrix.reshape(3,3);

        double const p[3] = {X, Y, Z};
        double result[2] = {0,0};
        const double focal[2] = {f_x, f_y};
        const double principal[2] = {p_x, p_y};

        const double R[9] = {1,0,0,   0,1,0,   0,0,1};

        const double t[3] = {t_x, t_y, t_z};

        cv::Mat_<double> mat_p = {X, Y, Z};
        mat_p = mat_p.reshape(3);
        cv::Mat_<double> mat_r = {R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8]};
        mat_r = mat_r.reshape(3,3);
        cv::Mat_<double> mat_t = {t_x, t_y, t_z};
        cv::Mat mat_result;

        cv::projectPoints(mat_p, mat_r, mat_t, camera_matrix, cv::noArray(), mat_result);

        cv::Mat_<double> mat_result2(mat_result);

        hdcalib::Calib::project(p, result, focal, principal, R, t);
        EXPECT_NEAR(result[0], mat_result2(0,0), 1e-8);
        EXPECT_NEAR(result[1], mat_result2(0,1), 1e-8);
    }
}

TEST(CalibrationResult, project_simple_rotate) {
    for (size_t ii = 0; ii < 3000; ++ii) {
        double const X = dist(engine), Y = dist(engine), Z = square_p1(dist(engine)),
                f_x = square_p1(dist(engine)), f_y = square_p1(dist(engine)),
                p_x = dist(engine), p_y = dist(engine),
                t_x = dist(engine), t_y = dist(engine), t_z = square_p1(dist(engine)),
                r_a = dist(engine), r_b = dist(engine), r_c = dist(engine);

        cv::Mat_<double> camera_matrix = {f_x,0,p_x,   0,f_y,p_y,   0,0,1};
        camera_matrix = camera_matrix.reshape(3,3);

        double const p[3] = {X, Y, Z};
        double result[2] = {0,0};
        const double focal[2] = {f_x, f_y};
        const double principal[2] = {p_x, p_y};

        const double rot[3] = {r_a, r_b, r_c};
        double R[9] = {1,0,0,   0,1,0,   0,0,1};
        hdcalib::Calib::rot_vec2mat(rot, R);

        const double t[3] = {t_x, t_y, t_z};

        cv::Mat_<double> mat_p = {X, Y, Z};
        mat_p = mat_p.reshape(3);
        cv::Mat_<double> mat_rot = {r_a, r_b, r_c};
        mat_rot = mat_rot.reshape(3);
        cv::Mat mat_r;
        cv::Rodrigues(mat_rot, mat_r);
        cv::Mat_<double> mat_t = {t_x, t_y, t_z};
        cv::Mat mat_result;

        cv::projectPoints(mat_p, mat_r, mat_t, camera_matrix, cv::noArray(), mat_result);

        cv::Mat_<double> mat_result2(mat_result);

        hdcalib::Calib::project(p, result, focal, principal, R, t);
        EXPECT_NEAR(result[0], mat_result2(0,0), 1e-8);
        EXPECT_NEAR(result[1], mat_result2(0,1), 1e-8);
    }
}

TEST(CalibrationResult, project_distorted_12) {
    for (size_t ii = 0; ii < 3000; ++ii) {
        double const X = dist(engine), Y = dist(engine), Z = square_p1(dist(engine)),
                f_x = square_p1(dist(engine)), f_y = square_p1(dist(engine)),
                p_x = dist(engine), p_y = dist(engine),
                t_x = dist(engine), t_y = dist(engine), t_z = square_p1(dist(engine)),
                r_a = dist(engine), r_b = dist(engine), r_c = dist(engine);

        double const d1 = dist(engine), d2 = dist(engine), d3 = dist(engine), d4 = dist(engine), d5 = dist(engine), d6 = dist(engine), d7 = dist(engine), d8 = dist(engine), d9 = dist(engine), d10 = dist(engine), d11 = dist(engine), d12 = dist(engine);

        double dist[14] = {d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,0,0};
        cv::Mat_<double> mat_dist = {d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,0,0};

        // Example dist coeffficients: 0.1137881268110966, -4.528681803108614, -0.004664257015838463, 0.02662626040742514, -32.1431298587843, -0.1532152732065023, -3.824127654198806, -29.8193079302472, -0.02567031702230461, -0.003698057248079691, 0.001242341955761071, 0.004512906522158435, -0.002542875752550364, 0.05120483952668687

        cv::Mat_<double> camera_matrix = {f_x,0,p_x,   0,f_y,p_y,   0,0,1};
        camera_matrix = camera_matrix.reshape(3,3);

        double const p[3] = {X, Y, Z};
        double result[2] = {0,0};
        const double focal[2] = {f_x, f_y};
        const double principal[2] = {p_x, p_y};

        const double rot[3] = {r_a, r_b, r_c};
        double R[9] = {1,0,0,   0,1,0,   0,0,1};
        hdcalib::Calib::rot_vec2mat(rot, R);

        const double t[3] = {t_x, t_y, t_z};

        cv::Mat_<double> mat_p = {X, Y, Z};
        mat_p = mat_p.reshape(3);
        cv::Mat_<double> mat_rot = {r_a, r_b, r_c};
        mat_rot = mat_rot.reshape(3);
        cv::Mat mat_r;
        cv::Rodrigues(mat_rot, mat_r);
        cv::Mat_<double> mat_t = {t_x, t_y, t_z};
        cv::Mat mat_result;

        cv::projectPoints(mat_p, mat_r, mat_t, camera_matrix, mat_dist, mat_result);

        cv::Mat_<double> mat_result2(mat_result);

        hdcalib::Calib::project(p, result, focal, principal, R, t, dist);
        EXPECT_TRUE(RelativeNear(result[0], mat_result2(0,0), 1e-8));
        EXPECT_TRUE(RelativeNear(result[1], mat_result2(0,1), 1e-8));
    }
}

TEST(CalibrationResult, project_distorted_14) {
    for (size_t ii = 0; ii < 3000; ++ii) {
        double const X = dist(engine), Y = dist(engine), Z = square_p1(dist(engine)),
                f_x = square_p1(dist(engine)), f_y = square_p1(dist(engine)),
                p_x = dist(engine), p_y = dist(engine),
                t_x = dist(engine), t_y = dist(engine), t_z = square_p1(dist(engine)),
                r_a = dist(engine), r_b = dist(engine), r_c = dist(engine);

        double const d1 = dist(engine), d2 = dist(engine), d3 = dist(engine), d4 = dist(engine), d5 = dist(engine), d6 = dist(engine), d7 = dist(engine), d8 = dist(engine), d9 = dist(engine), d10 = dist(engine), d11 = dist(engine), d12 = dist(engine), d13 = dist(engine), d14 = dist(engine);

        double dist[14] = {d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14};
        cv::Mat_<double> mat_dist = {d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14};

        // Example dist coeffficients: 0.1137881268110966, -4.528681803108614, -0.004664257015838463, 0.02662626040742514, -32.1431298587843, -0.1532152732065023, -3.824127654198806, -29.8193079302472, -0.02567031702230461, -0.003698057248079691, 0.001242341955761071, 0.004512906522158435, -0.002542875752550364, 0.05120483952668687

        cv::Mat_<double> camera_matrix = {f_x,0,p_x,   0,f_y,p_y,   0,0,1};
        camera_matrix = camera_matrix.reshape(3,3);

        double const p[3] = {X, Y, Z};
        double result[2] = {0,0};
        const double focal[2] = {f_x, f_y};
        const double principal[2] = {p_x, p_y};

        const double rot[3] = {r_a, r_b, r_c};
        double R[9] = {1,0,0,   0,1,0,   0,0,1};
        hdcalib::Calib::rot_vec2mat(rot, R);

        const double t[3] = {t_x, t_y, t_z};

        cv::Mat_<double> mat_p = {X, Y, Z};
        mat_p = mat_p.reshape(3);
        cv::Mat_<double> mat_rot = {r_a, r_b, r_c};
        mat_rot = mat_rot.reshape(3);
        cv::Mat mat_r;
        cv::Rodrigues(mat_rot, mat_r);
        cv::Mat_<double> mat_t = {t_x, t_y, t_z};
        cv::Mat mat_result;

        cv::projectPoints(mat_p, mat_r, mat_t, camera_matrix, mat_dist, mat_result);

        cv::Mat_<double> mat_result2(mat_result);

        hdcalib::Calib::project(p, result, focal, principal, R, t, dist);
        EXPECT_TRUE(RelativeNear(result[0], mat_result2(0,0), 1e-10));
        EXPECT_TRUE(RelativeNear(result[1], mat_result2(0,1), 1e-10));
    }
}

TEST(CalibrationResult, rot_vec2mat) {
    for (size_t ii = 0; ii < 10000; ++ii) {
        double const r_a = dist(engine), r_b = dist(engine), r_c = dist(engine);

        double r[3] = {r_a, r_b, r_c};
        double result[9];


        cv::Mat_<double> mat_r = {r_a, r_b, r_c};

        cv::Mat mat_result;
        cv::Rodrigues(mat_r, mat_result);


        cv::Mat_<double> mat_result2(mat_result);

        hdcalib::Calib::rot_vec2mat(r, result);
        EXPECT_NEAR(result[0], mat_result2(0,0), 1e-8);
        EXPECT_NEAR(result[1], mat_result2(0,1), 1e-8);
        EXPECT_NEAR(result[2], mat_result2(0,2), 1e-8);

        EXPECT_NEAR(result[3], mat_result2(1,0), 1e-8);
        EXPECT_NEAR(result[4], mat_result2(1,1), 1e-8);
        EXPECT_NEAR(result[5], mat_result2(1,2), 1e-8);

        EXPECT_NEAR(result[6], mat_result2(2,0), 1e-8);
        EXPECT_NEAR(result[7], mat_result2(2,1), 1e-8);
        EXPECT_NEAR(result[8], mat_result2(2,2), 1e-8);
    }
}

template<class T1, class T2, class T3>
void printArrays(std::vector<T1> const& a, std::vector<T2> const& b, std::vector<T3> const& c) {
    for (size_t ii = 0; ii < a.size() && ii < b.size() && ii < c.size(); ++ii) {
        std::cout << a[ii] << "\t" << b[ii] << "\t" << c[ii] << std::endl;
    }
    std::cout << std::endl;
}

TEST(Calib, insertSorted) {
    std::vector<std::string> a = {"a", "b", "z", "c"};
    std::vector<std::string> b = {"1", "2", "26", "3"};
    std::vector<std::string> c = {"alpha", "beta", "omega", "gamma"};

    hdcalib::Calib::insertSorted(a, b, c);

    printArrays(a,b,c);

    EXPECT_EQ(a[0], "a");
    EXPECT_EQ(a[1], "b");
    EXPECT_EQ(a[2], "c");
    EXPECT_EQ(a[3], "z");

    EXPECT_EQ(b[0], "1");
    EXPECT_EQ(b[1], "2");
    EXPECT_EQ(b[2], "3");
    EXPECT_EQ(b[3], "26");

    EXPECT_EQ(c[0], "alpha");
    EXPECT_EQ(c[1], "beta");
    EXPECT_EQ(c[2], "gamma");
    EXPECT_EQ(c[3], "omega");
}

std::string random_string( size_t length ) {
    auto randchar = []() -> char
    {
            const char charset[] =
            "0123456789"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz";
            const size_t max_index = (sizeof(charset) - 1);
            return charset[ size_t(rand()) % max_index ];
};
std::string str(length,0);
std::generate_n( str.begin(), length, randchar );
return str;
}

TEST(Calib, insertSortedRandom) {
    for (size_t maxlength = 0; maxlength < 10; ++maxlength) {
        for (size_t jj = 0; jj < 50; ++jj) {

            std::vector<std::string> a;
            std::vector<std::string> b;
            std::vector<std::string> c;

            size_t stringlength = 3;

            for (size_t ii = 1; ii < maxlength; ++ii) {
                a.push_back(random_string(stringlength));
                b.push_back(random_string(stringlength));
                c.push_back(random_string(stringlength));
            }

            std::sort(a.begin(), a.end());
            a.push_back(random_string(stringlength));
            b.push_back(random_string(stringlength));
            c.push_back(random_string(stringlength));

            std::map<std::string, std::pair<std::string, std::string> > map;
            for (size_t ii = 0; ii < a.size(); ++ii) {
                map[a[ii]] = std::pair<std::string, std::string>(b[ii], c[ii]);
            }

            hdcalib::Calib::insertSorted(a, b, c);

            for (size_t ii = 0; ii < a.size(); ++ii) {
                auto const& it = map.find(a[ii]);
                EXPECT_NE(it, map.end());
                if (it != map.end()) {
                    EXPECT_EQ(it->second.first, b[ii]);
                    EXPECT_EQ(it->second.second, c[ii]);
                }
            }
            for (size_t ii = 1; ii < a.size(); ++ii) {
                EXPECT_TRUE(a[ii-1] < a[ii]);
            }
        }
    }

}

TEST(Calib, get3DPoint) {
    //Vec3d Calib::get3DPoint(const Corner &c, const Mat &_rvec, const Mat &_tvec) {

    hdcalib::CornerStore store;
    getCornerGrid(store, 3, 3);
    hdcalib::Calib c;
    c.addInputImage("test1", store);

    cv::Mat_<double> rvec(3,1);
    cv::Mat_<double> tvec(3,1);

    for (size_t ii = 0; ii < 3; ++ii) {
        rvec(int(ii)) = 0;
        tvec(int(ii)) = ii;
    }

    store = c.get("test1");
    std::vector<hdcalib::Corner> corners = store.getCorners();

    for (hdmarker::Corner const& corner : corners) {
        cv::Vec3d point = c.get3DPoint(corner, rvec, tvec);
        std::cout << "Corner: " << corner.id << ", " << corner.page << std::endl
                  << "Position: " << point << std::endl << std::endl;
    }
    std::cout << std::endl;
}

void printMarkers(std::vector<hdmarker::Corner> const& corners) {
    for (size_t ii = 0; ii < corners.size(); ++ii) {
        std::cout << ii << "\t" << corners[ii].id << ", " << corners[ii].page << std::endl;
    }
    std::cout << std::endl;
}

TEST(Calib, rot4_rot3) {
    using namespace hdcalib;

    runningstats::RunningStats diff_stats[3];
    std::discrete_distribution<int> zero_dice {1,1};

    for (size_t jj = 0; jj < 10*1000; ++jj) {

        double const src[3] = {dist(engine), dist(engine), dist(engine)};
        double const rotation[3] = {zero_dice(engine)*dist(engine),
                                    zero_dice(engine)*dist(engine),
                                    zero_dice(engine)*dist(engine)};
        double rot3_mat[9];
        double translation[3] = {0,0,0};
        Calib::rot_vec2mat(rotation, rot3_mat);
        double rot3_result[3];
        Calib::get3DPoint(src, rot3_result, rot3_mat, translation);

        double rot4_mat[9];
        double rot4_vec[4];
        Calib::rot3_rot4(rotation, rot4_vec);
        Calib::rot4_vec2mat(rot4_vec, rot4_mat);
        double rot4_result[3];
        Calib::get3DPoint(src, rot4_result, rot4_mat, translation);

        for (size_t ii = 0; ii < 3; ++ii) {
            diff_stats[ii].push(rot3_result[ii] - rot4_result[ii]);
            EXPECT_NEAR(rot3_result[ii], rot4_result[ii], 1e-12);
        }
    }

    std::cout << "rot3 - rot4 result stats:" << std::endl;
    for (size_t ii = 0; ii < 3; ++ii) {
        std::cout << diff_stats[ii].print() << std::endl;
    }

}

TEST(Calib, rot4_rot3_backwards) {
    using namespace hdcalib;

    runningstats::RunningStats diff_stats[3];

    std::discrete_distribution<int> zero_dice {1,1};

    for (size_t jj = 0; jj < 10*1000; ++jj) {

        double const src[3] = {dist(engine), dist(engine), dist(engine)};
        double const rotation[4] = {zero_dice(engine)*dist(engine),
                                    zero_dice(engine)*dist(engine),
                                    zero_dice(engine)*dist(engine),
                                    zero_dice(engine)*dist(engine)};
        double rot3_mat[9];
        double rot3_vec[3];
        double translation[3] = {0,0,0};
        Calib::rot4_rot3(rotation, rot3_vec);
        Calib::rot_vec2mat(rot3_vec, rot3_mat);
        double rot3_result[3];
        Calib::get3DPoint(src, rot3_result, rot3_mat, translation);

        double rot4_mat[9];
        Calib::rot4_vec2mat(rotation, rot4_mat);
        double rot4_result[3];
        Calib::get3DPoint(src, rot4_result, rot4_mat, translation);

        bool has_error = false;
        for (size_t ii = 0; ii < 3; ++ii) {
            diff_stats[ii].push(rot3_result[ii] - rot4_result[ii]);
            EXPECT_NEAR(rot3_result[ii], rot4_result[ii], 1e-12);
            if (std::abs(rot3_result[ii] - rot4_result[ii]) > 1e-12) {
                has_error = true;
            }
        }
        if (has_error) {
            std::cout << "rotation: " << rotation[0] << ", " << rotation[1] << ", " << rotation[2] << ", " << rotation[3] << std::endl;
            Calib::rot4_rot3(rotation, rot3_vec);
            Calib::rot4_vec2mat(rotation, rot4_mat);
        }
    }

    std::cout << "rot3 - rot4 backwards result stats:" << std::endl;
    for (size_t ii = 0; ii < 3; ++ii) {
        std::cout << diff_stats[ii].print() << std::endl;
    }

}

TEST(Calib, rot3_rot4_rot3_conversion) {
    using namespace hdcalib;

    runningstats::RunningStats diff_stats[3];

    std::discrete_distribution<int> zero_dice {1,1};

    for (size_t jj = 0; jj < 10*1000; ++jj) {

        double const rotation[3] = {zero_dice(engine)*dist(engine),
                                    zero_dice(engine)*dist(engine),
                                    zero_dice(engine)*dist(engine)};
        cv::Mat_<double> const rvec3_src = {rotation[0], rotation[1], rotation[2]};
        std::vector<double> intermediate = Calib::rot3_rot4<double>(rvec3_src);
        EXPECT_EQ(4, intermediate.size());

        cv::Mat_<double> rvec3_dst;
        Calib::rot4_rot3(intermediate.data(), rvec3_dst);
        for (int ii = 0; ii < 3; ++ii) {
            EXPECT_NEAR(rvec3_src(ii), rvec3_dst(ii), 1e-12);
            diff_stats[ii].push_unsafe(rvec3_src(ii) - rvec3_dst(ii));
        }

    }

    std::cout << "rot3 - rot4 - rot3 result stats:" << std::endl;
    for (size_t ii = 0; ii < 3; ++ii) {
        std::cout << diff_stats[ii].print() << std::endl;
    }

}

bool rot4_zero(double const vec[4]) {
    return std::abs(vec[3]) < 100*std::numeric_limits<double>::min();
}

TEST(Calib, rot4_rot3_rot4_conversion) {
    using namespace hdcalib;

    runningstats::RunningStats diff_stats[4];

    std::discrete_distribution<int> zero_dice {1,1};

    for (size_t jj = 0; jj < 10*1000; ++jj) {

        double rotation[4] = {zero_dice(engine)*dist(engine),
                                    zero_dice(engine)*dist(engine),
                                    zero_dice(engine)*dist(engine),
                                    zero_dice(engine)*dist(engine)};

        Calib::normalize_rot4(rotation, rotation);

        cv::Mat_<double> intermediate;
        Calib::rot4_rot3<double>(rotation, intermediate);

        double rvec4_dst[4];
        Calib::rot3_rot4(intermediate, rvec4_dst);

        bool failed = false;
        EXPECT_TRUE(rot4_eq(rotation, rvec4_dst));
        if (!rot4_zero(rotation)) {
            for (size_t ii = 0; ii < 4; ++ii) {
                diff_stats[ii].push_unsafe(rotation[ii] - rvec4_dst[ii]);
                if (std::abs(rotation[ii] - rvec4_dst[ii]) > 1e-12) {
                    failed = true;
                }
            }
        }
        if (failed) {
            Calib::rot4_rot3<double>(rotation, intermediate);
            Calib::rot3_rot4(intermediate, rvec4_dst);
        }

    }

    std::cout << "rot4 - rot3 - rot4 result stats:" << std::endl;
    for (size_t ii = 0; ii < 4; ++ii) {
        std::cout << diff_stats[ii].print() << std::endl;
    }

}

#include "cornercolor.h"

TEST(CornerColor, all) {
    using namespace hdcalib;

    // Test the main markers at recursion levels 0-3
    for (int rec = 0; rec <= 3; ++rec) {
        int const factor = Calib::computeCornerIdFactor(rec);
        for (int ii = 0; ii < 32; ++ii) {
            EXPECT_EQ(2, CornerColor::getColor(factor*cv::Point2i(ii,ii), 0, rec));
            EXPECT_EQ(2, CornerColor::getColor(factor*cv::Point2i(ii,ii % 2), 0, rec));
            EXPECT_EQ(2, CornerColor::getColor(factor*cv::Point2i(ii % 2, ii), 0, rec));
        }
        for (int ii = 0; ii < 31; ++ii) {
            EXPECT_EQ(3, CornerColor::getColor(factor*cv::Point2i(ii+1,ii), 0, rec));
            EXPECT_EQ(3, CornerColor::getColor(factor*cv::Point2i(ii,ii+1), 0, rec));
            EXPECT_EQ(3, CornerColor::getColor(factor*cv::Point2i(ii+1,ii % 2), 0, rec));
            EXPECT_EQ(3, CornerColor::getColor(factor*cv::Point2i(ii % 2, ii+1), 0, rec));
        }
    }

    // Test the submarkers at the fringe at recursion level 1.
    for (int main = 0; main < 32; ++main) {
        for (int xx : {1,3,5,7,9}) {
            EXPECT_EQ((main+1) % 2, CornerColor::getColor({10*main + xx,1}, 0, 1));
        }
    }

    // Test the submarkers at the fringe at recursion levels 2-3.
    for (int rec = 1; rec <= 3; ++rec) {
        int const factor = Calib::computeCornerIdFactor(rec);
        for (int main = 0; main < 32; ++main) {
            for (int xx = 1; xx < factor; xx += 2) {
                for (int fringe = 1; fringe < 2*rec && fringe < 4; fringe += 2) {
                    EXPECT_EQ((main+1) % 2, CornerColor::getColor({factor*main + xx,fringe}, 0, rec));
                    if ((main+1) % 2 != CornerColor::getColor({factor*main + xx,fringe}, 0, rec)) {
                        std::cout << "Failed rec: " << rec
                                  << ", main: " << main
                                  << ", x: " << xx
                                  << ", id: " << cv::Point2i{factor*main + xx,fringe}
                                  << ", got " << CornerColor::getColor({factor*main + xx,1}, 0, rec)
                                  << ", expected " << (main+1) % 2 << std::endl;
                    }
                }
            }
        }
    }

    // Test a couple of black submarkers at level 1 by hand.
    for (cv::Point2i const & id : {
         cv::Point2i{7,3},
         cv::Point2i{5,5},
         cv::Point2i{9,5},
         cv::Point2i{3,7},
         cv::Point2i{7,7},

         cv::Point2i{13,5},
         cv::Point2i{15,5},
         cv::Point2i{17,5},
}) {
        EXPECT_EQ(0, CornerColor::getColor(id, 0, 1));
        if (0 != CornerColor::getColor(id, 0, 1)) {
            std::cout << "Failed " << id
                      << ", got " << CornerColor::getColor(id, 0, 1)
                      << ", expected 0" << std::endl;
        }
    }

    // Test a couple of white submarkers at level 1 by hand.
    for (cv::Point2i const & id : {
         cv::Point2i{3,3},
         cv::Point2i{5,3},
         cv::Point2i{3,5},
         cv::Point2i{9,3},
         cv::Point2i{7,5},

         cv::Point2i{5,7},
         cv::Point2i{9,7},
         cv::Point2i{3,9},
         cv::Point2i{5,9},
         cv::Point2i{7,9},
         cv::Point2i{9,9},
}) {
        EXPECT_EQ(1, CornerColor::getColor(id, 0, 1));
        if (1 != CornerColor::getColor(id, 0, 1)) {
            std::cout << "Failed " << id
                      << ", got " << CornerColor::getColor(id, 0, 1)
                      << ", expected 1" << std::endl;
        }
    }


    // Test a couple of black submarkers at level 2 by hand.
    for (cv::Point2i const & id : {
         cv::Point2i{5,5},
         cv::Point2i{5,15},
         cv::Point2i{15,5},
         cv::Point2i{25,5},
         cv::Point2i{5,25},
         cv::Point2i{15,25},
         cv::Point2i{25,15},
         cv::Point2i{5,35},
         cv::Point2i{35,5},
         cv::Point2i{45,5},
         cv::Point2i{5,45},

         cv::Point2i{31,11},
         cv::Point2i{31,13},
         cv::Point2i{31,17},
         cv::Point2i{31,19},

         cv::Point2i{33,11},
         cv::Point2i{33,13},
         cv::Point2i{33,17},
         cv::Point2i{33,19},

         cv::Point2i{35,11},
         cv::Point2i{35,13},
         cv::Point2i{35,17},
         cv::Point2i{35,19},

         cv::Point2i{37,11},
         cv::Point2i{37,13},
         cv::Point2i{37,17},
         cv::Point2i{37,19},

         cv::Point2i{39,11},
         cv::Point2i{39,13},
         cv::Point2i{39,17},
         cv::Point2i{39,19},

}) {
        EXPECT_EQ(0, CornerColor::getColor(id, 0, 2));
        if (0 != CornerColor::getColor(id, 0, 2)) {
            std::cout << "failed " << id << std::endl;
        }
    }
    std::cout << "Num calls: " << CornerColor::getNumCalls() << std::endl;
}


std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = uchar(1 + (type >> CV_CN_SHIFT));

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += char(chans+'0');

  return r;
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    std::cout << "RUN_ALL_TESTS return value: " << RUN_ALL_TESTS() << std::endl;
}
