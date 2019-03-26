#include "hdcalib.h"
#include <gtest/gtest.h>

void getCornerGrid(
        hdcalib::CornerStore & store,
        size_t const grid_width = 50,
        size_t const grid_height = 50,
        int const page = 0,
        cv::Point2f offset = cv::Point2f(0,0)) {
    hdmarker::Corner a;
    for (size_t ii = 0; ii < grid_width; ++ii) {
        for (size_t jj = 0; jj < grid_height; ++jj) {
            a.id = cv::Point2i(ii, jj);
            a.page = page;
            a.p = cv::Point2f(ii, jj) + offset;
            a.size = .5;
            store.push_back(a);
        }
    }
}

int main(void) {
    {
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
    {
        hdcalib::CornerStore s;
        getCornerGrid(s, 5, 5, 0);

        hdmarker::Corner c(cv::Point2f(2,2), cv::Point2i(2,2), 0);

        if (s.hasID(c)) {
            std::cout << "Found corner!" << std::endl;
        }

        std::vector<hdmarker::Corner> res = s.findByPos(c, 5);

        std::cout << "Found corners in a neighbourhood of " << c.p << std::endl;
        for (auto const& f : res) {
            std::cout << "p: " << f.p << ",\t id: " << f.id << ",\t page: " << f.page << std::endl;
        }
    }
}
