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

}

int main(int argc, char** argv)
{
    /* Use this code if the tests fail with unexpected exceptions.
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

    return 0;
    // */

    testing::InitGoogleTest(&argc, argv);
    std::cout << "RUN_ALL_TESTS return value: " << RUN_ALL_TESTS() << std::endl;

}
