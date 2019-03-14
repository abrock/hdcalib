#include <iostream>

#include <gtest/gtest.h>

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <algorithm>
#include <random>

#include "hdmarker.hpp"
#include "hdcalib.h"

std::random_device rd;
std::default_random_engine engine(rd());
std::normal_distribution<double> dist;

void getCornerGrid(
        hdcalib::CornerStore & store,
        size_t const grid_width = 50,
        size_t const grid_height = 50) {
    hdmarker::Corner a;
    for (size_t ii = 0; ii < grid_width; ++ii) {
        for (size_t jj = 0; jj < grid_height; ++jj) {
            a.id = cv::Point2i(ii, jj);
            a.page = 0;
            a.p = cv::Point2f(ii, jj);
            a.size = .5;
            store.push_back(a);
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
    if (0 == a || 0 == b) {
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
        x.id = cv::Point2i(ii*12+3,ii*12+4);
        x.pc[0] = cv::Point2f(ii*12+5,ii*12+6);
        x.pc[1] = cv::Point2f(ii*12+7,ii*12+8);
        x.pc[2] = cv::Point2f(ii*12+9,ii*12+10);
        x.page = ii*12+11;
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
            a.id = cv::Point2i(ii, jj);
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

    std::uniform_real_distribution<float> dist_x(-.4, grid_width - .6);
    std::uniform_real_distribution<float> dist_y(-.4, grid_height - .6);
    for (size_t ii = 0; ii < 10*1000; ++ii) {
        float const x = dist_x(engine);
        float const y = dist_y(engine);
        cv::Point2i id(std::round(x), std::round(y));
        std::vector<hdmarker::Corner> res = store.findByPos(x, y, 1);
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
    store.purgeUnlikely();
    EXPECT_EQ(store.size(), old_size);

    for (size_t ii = 0; ii < grid_width; ++ii) {
        for (size_t jj = 0; jj < grid_height; ++jj) {
            a.id = cv::Point2i(ii, jj);
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
        cv::Point2i id(std::round(x), std::round(y));
        std::vector<hdmarker::Corner> res = store.findByPos(x, y, 1);
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

int main(int argc, char** argv)
{
    //* Use this code if the tests fail with unexpected exceptions.
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

    store.push_back(a);

    getCornerGrid(store, 10, 10);

    store.purgeUnlikely();

    //return 0;
    // */

    testing::InitGoogleTest(&argc, argv);
    std::cout << "RUN_ALL_TESTS return value: " << RUN_ALL_TESTS() << std::endl;

}
